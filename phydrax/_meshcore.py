#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host binding of the native Phydrax meshcore library.

The optional ``phydrax-meshcore`` distribution ships a C++ shared library with a
plain C ABI (``native/meshcore/include/phydrax_meshcore.h``) that is loaded with
``ctypes``. It owns geometric predicates (Shewchuk adaptive expansion
arithmetic with index-ordered symbolic perturbation), exactly classified convex
clipping moments, contact classification of triangles and segments in
3D, -predicate Delaunay/regular/constrained
triangulations, and the deterministic multilevel graph partition backend of
:mod:`phydrax.graph`. Every function here is host-only NumPy interchange: inputs are
validated and converted to contiguous float64/int32 arrays, results are fresh
arrays, and native statuses reach the caller (per-item status arrays for batched
clipping, exceptions for rejected calls).

Library lookup order: the native ``ProviderConfig.library_paths`` override,
then ``PHYDRAX_MESHCORE_LIBRARY``, then ``phydrax_meshcore.library_path()``. A
library is usable only when its exported ``phx_mc_abi_contract`` equals this
binding's C ABI contract (verified before any other entry point is bound), it
exports every C ABI symbol bound here, its build configuration declares the
required independent request/workflow clock capability, and its version equals
the installed ``phydrax`` release; anything else is reported as
:class:`MeshcoreUnavailableError` with the reason.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import math
import threading
import time
import weakref
from _ctypes import CFuncPtr
from collections import ChainMap
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import AbstractContextManager, contextmanager, ExitStack
from contextvars import ContextVar, Token
from enum import IntEnum
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    assert_never,
    cast,
    final,
    Literal,
    NamedTuple,
    NoReturn,
    overload,
    Protocol,
    TypeAlias,
)

import equinox as eqx
import numpy as np
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray

from ._config_environment import LIBRARY_ENVIRONMENT
from ._config_state import library_input
from ._host_storage import retained_host_storage_profile
from ._resource_ticket import BatchTicket, ManagedBytesContributor
from ._strict import StrictModule
from ._validation import bounded_integer, nonnegative_integer
from .typing import checked, ConvertibleToArray


# SHA-256 of ``native/meshcore/include/phydrax_meshcore.h`` with CRLF normalized
# to LF: the public C declarations that ``_SIGNATURES`` binds. The native build
# exports the same digest of the header it compiled.
_ABI_CONTRACT = "f9cf043d91e2574194620fae38b6843987305a494c93d0cb31917d7669c4ed06"
_EXECUTION_CLOCK_CAPABILITY = "execution_clocks=request-preparation+absolute-workflow"
_MAX_POINTS = 2147483646
_MAX_COUNTER_VALUE = (1 << 64) - 1


def _input_summary(value: object, /) -> str:
    """Summarize a refused host input without copying or reading array payloads."""
    match value:
        case np.ndarray():
            return f"shape={value.shape}, dtype={value.dtype}"
        case None | bool() | int() | float() | str() | np.generic():
            return repr(value)
        case tuple() | list():
            return f"{type(value).__qualname__}(length={len(value)})"
        case _:
            return type(value).__qualname__


class MeshcoreUnavailableError(ImportError):
    """The native meshcore library is not installed or cannot be loaded."""


class MeshcoreError(RuntimeError):
    """The native meshcore library refused a call for resource or internal reasons."""

    def __init__(
        self,
        status: MeshcoreStatus,
        message: str,
        /,
        *,
        work_counters: NativeExecutionWorkCounters | None = None,
        memory_counters: NativeExecutionMemoryCounters | None = None,
        elapsed_seconds: float | None = None,
        maximum_seconds: float | None = None,
    ) -> None:
        super().__init__(f"{message} (meshcore status {status.name})")
        self.status = status
        self.work_counters = work_counters
        self.memory_counters = memory_counters
        self.elapsed_seconds = elapsed_seconds
        self.maximum_seconds = maximum_seconds
        self.batch_admitted_upper: tuple[int, int, int, int] | None = None
        self.batch_requested_upper: tuple[int, int, int, int] | None = None
        self.batch_refusal_evidence: NDArray[np.uint64] | None = None


class MeshcoreStatus(IntEnum):
    """Call and item status codes of the meshcore C ABI."""

    OK = 0
    INVALID_ARGUMENT = 1
    NONFINITE_INPUT = 2
    RANGE_ERROR = 3
    DEGENERATE_INPUT = 4
    INVALID_INPUT = 5
    CAPACITY_EXCEEDED = 6
    CONSTRAINT_INTERSECTION = 7
    REFINEMENT_LIMIT = 8
    INTERNAL_ERROR = 9
    TIMEOUT = 10


class QuadraticTetrahedronValidityDecision(IntEnum):
    """Exact root result codes returned by quadratic tetrahedron validity."""

    NEEDS_SUBDIVISION = 0
    ROOT_VALID = 1
    ROOT_INVALID = 2
    BERNSTEIN_NODE_LIMIT = 3
    DEPTH_ZERO_UNRESOLVED = 4


@final
class QuadraticTetrahedronValidityResult(StrictModule):
    """Private root decisions and outward bounds from the native exact kernel."""

    decisions: NDArray[np.int32]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    evidence: NDArray[np.int64]

    def __init__(
        self,
        decisions: NDArray[np.int32],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        evidence: NDArray[np.int64],
        /,
    ) -> None:
        if not isinstance(decisions, np.ndarray):
            raise TypeError("QuadraticTetrahedronValidityResult.decisions must be a NumPy array.")
        if decisions.ndim != 1:
            raise ValueError("QuadraticTetrahedronValidityResult.decisions must have rank one.")
        count = decisions.shape[0]
        for name, value, dtype, shape in (
            ("decisions", decisions, np.int32, (count,)),
            ("lower", lower, np.float64, (count,)),
            ("upper", upper, np.float64, (count,)),
            ("evidence", evidence, np.int64, (6,)),
        ):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"QuadraticTetrahedronValidityResult.{name} must be a NumPy array.")
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise ValueError(
                    f"QuadraticTetrahedronValidityResult.{name} must have shape {shape} and dtype {np.dtype(dtype)}."
                )
        if (
            any(evidence[index] < 0 for index in range(3))
            or evidence[3] != count
            or evidence[4] != -1
            or evidence[5] != 0
        ):
            raise ValueError("QuadraticTetrahedronValidityResult.evidence must describe complete success.")
        for cell, code in enumerate(decisions):
            decision = QuadraticTetrahedronValidityDecision(int(code))
            if decision is QuadraticTetrahedronValidityDecision.BERNSTEIN_NODE_LIMIT:
                if not (math.isnan(float(lower[cell])) and math.isnan(float(upper[cell]))):
                    raise ValueError("Bernstein-node refusal must retain unavailable determinant bounds.")
            elif math.isnan(float(lower[cell])) or math.isnan(float(upper[cell])) or lower[cell] > upper[cell]:
                raise ValueError("Quadratic tetrahedron determinant bounds must be ordered and non-NaN.")
        for value in (decisions, lower, upper, evidence):
            value.flags.writeable = False
        self.decisions = decisions
        self.lower = lower
        self.upper = upper
        self.evidence = evidence


@final
class QuadraticTetrahedronValidityError(MeshcoreError):
    """Native execution refusal with its always-published six-slot evidence."""

    evidence: NDArray[np.int64]

    def __init__(
        self,
        status: MeshcoreStatus,
        evidence: NDArray[np.int64],
        /,
    ) -> None:
        if evidence.shape != (6,) or evidence.dtype != np.dtype(np.int64):
            raise ValueError(
                "QuadraticTetrahedronValidityError.evidence must have shape (6,) and dtype int64."
            )
        evidence.flags.writeable = False
        super().__init__(
            status,
            "quadratic_tetrahedron_validity: native exact root evaluation failed",
        )
        self.evidence = evidence


@final
class GraphPartitionCounters(StrictModule):
    """Named graph partition counters from the native ABI slots."""

    coarsening_levels: int = eqx.field(static=True, default=0)
    coarsest_vertex_count: int = eqx.field(static=True, default=0)
    matched_pairs: int = eqx.field(static=True, default=0)
    bisection_trials: int = eqx.field(static=True, default=0)
    refinement_passes: int = eqx.field(static=True, default=0)
    moves_committed: int = eqx.field(static=True, default=0)
    moves_rolled_back: int = eqx.field(static=True, default=0)
    balance_moves: int = eqx.field(static=True, default=0)
    nonempty_repairs: int = eqx.field(static=True, default=0)
    adjacency_visits: int = eqx.field(static=True, default=0)
    candidate_evaluations: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> GraphPartitionCounters:
        if values.shape != (11,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "GraphPartitionCounters.slots must have shape (11,) and dtype int64; got incompatible native slots"
            )
        return cls(
            coarsening_levels=int(values[0]),
            coarsest_vertex_count=int(values[1]),
            matched_pairs=int(values[2]),
            bisection_trials=int(values[3]),
            refinement_passes=int(values[4]),
            moves_committed=int(values[5]),
            moves_rolled_back=int(values[6]),
            balance_moves=int(values[7]),
            nonempty_repairs=int(values[8]),
            adjacency_visits=int(values[9]),
            candidate_evaluations=int(values[10]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("coarsening_levels", self.coarsening_levels),
            ("coarsest_vertex_count", self.coarsest_vertex_count),
            ("matched_pairs", self.matched_pairs),
            ("bisection_trials", self.bisection_trials),
            ("refinement_passes", self.refinement_passes),
            ("moves_committed", self.moves_committed),
            ("moves_rolled_back", self.moves_rolled_back),
            ("balance_moves", self.balance_moves),
            ("nonempty_repairs", self.nonempty_repairs),
            ("adjacency_visits", self.adjacency_visits),
            ("candidate_evaluations", self.candidate_evaluations),
        )


@final
class SurfaceReconnectionCounters(StrictModule):
    """Named surface counters from the native ABI slots."""

    inserted: int = eqx.field(static=True, default=0)
    refused: int = eqx.field(static=True, default=0)
    flips: int = eqx.field(static=True, default=0)
    walk_steps: int = eqx.field(static=True, default=0)
    work: int = eqx.field(static=True, default=0)
    exhausted: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> SurfaceReconnectionCounters:
        if values.shape != (6,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "SurfaceReconnectionCounters.slots must have shape (6,) and dtype int64; got incompatible native slots"
            )
        return cls(
            inserted=int(values[0]),
            refused=int(values[1]),
            flips=int(values[2]),
            walk_steps=int(values[3]),
            work=int(values[4]),
            exhausted=int(values[5]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("inserted", self.inserted),
            ("refused", self.refused),
            ("flips", self.flips),
            ("walk_steps", self.walk_steps),
            ("work", self.work),
            ("exhausted", self.exhausted),
        )


@final
class TetMeshRefineCounters(StrictModule):
    """Named tet mesh refine counters from the native ABI slots."""

    circumcenter_insertions: int = eqx.field(static=True, default=0)
    subfacet_splits: int = eqx.field(static=True, default=0)
    subsegment_splits: int = eqx.field(static=True, default=0)
    encroachment_deferrals: int = eqx.field(static=True, default=0)
    refused_insertions: int = eqx.field(static=True, default=0)
    queue_pops: int = eqx.field(static=True, default=0)
    unmet_cells: int = eqx.field(static=True, default=0)
    work_units: int = eqx.field(static=True, default=0)
    largest_cavity: int = eqx.field(static=True, default=0)
    vertices: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> TetMeshRefineCounters:
        if values.shape != (10,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "TetMeshRefineCounters.slots must have shape (10,) and dtype int64; got incompatible native slots"
            )
        return cls(
            circumcenter_insertions=int(values[0]),
            subfacet_splits=int(values[1]),
            subsegment_splits=int(values[2]),
            encroachment_deferrals=int(values[3]),
            refused_insertions=int(values[4]),
            queue_pops=int(values[5]),
            unmet_cells=int(values[6]),
            work_units=int(values[7]),
            largest_cavity=int(values[8]),
            vertices=int(values[9]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("circumcenter_insertions", self.circumcenter_insertions),
            ("subfacet_splits", self.subfacet_splits),
            ("subsegment_splits", self.subsegment_splits),
            ("encroachment_deferrals", self.encroachment_deferrals),
            ("refused_insertions", self.refused_insertions),
            ("queue_pops", self.queue_pops),
            ("unmet_cells", self.unmet_cells),
            ("work_units", self.work_units),
            ("largest_cavity", self.largest_cavity),
            ("vertices", self.vertices),
        )

    def accumulated(self, other: TetMeshRefineCounters, /) -> TetMeshRefineCounters:
        """Accumulate work, retain final state counts and the largest cavity."""
        return TetMeshRefineCounters(
            circumcenter_insertions=self.circumcenter_insertions
            + other.circumcenter_insertions,
            subfacet_splits=self.subfacet_splits + other.subfacet_splits,
            subsegment_splits=self.subsegment_splits + other.subsegment_splits,
            encroachment_deferrals=self.encroachment_deferrals
            + other.encroachment_deferrals,
            refused_insertions=self.refused_insertions + other.refused_insertions,
            queue_pops=self.queue_pops + other.queue_pops,
            unmet_cells=other.unmet_cells,
            work_units=self.work_units + other.work_units,
            largest_cavity=max(self.largest_cavity, other.largest_cavity),
            vertices=other.vertices,
        )


@final
class TetMeshImproveCounters(StrictModule):
    """Named tet mesh improve counters from the native ABI slots."""

    face_removals: int = eqx.field(static=True, default=0)
    edge_removals: int = eqx.field(static=True, default=0)
    relocations: int = eqx.field(static=True, default=0)
    vertex_removals: int = eqx.field(static=True, default=0)
    passes: int = eqx.field(static=True, default=0)
    attempts: int = eqx.field(static=True, default=0)
    remaining_slivers: int = eqx.field(static=True, default=0)
    work_units: int = eqx.field(static=True, default=0)
    vertex_insertions: int = eqx.field(static=True, default=0)
    multiface_removals: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> TetMeshImproveCounters:
        if values.shape != (10,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "TetMeshImproveCounters.slots must have shape (10,) and dtype int64; got incompatible native slots"
            )
        return cls(
            face_removals=int(values[0]),
            edge_removals=int(values[1]),
            relocations=int(values[2]),
            vertex_removals=int(values[3]),
            passes=int(values[4]),
            attempts=int(values[5]),
            remaining_slivers=int(values[6]),
            work_units=int(values[7]),
            vertex_insertions=int(values[8]),
            multiface_removals=int(values[9]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("face_removals", self.face_removals),
            ("edge_removals", self.edge_removals),
            ("relocations", self.relocations),
            ("vertex_removals", self.vertex_removals),
            ("passes", self.passes),
            ("attempts", self.attempts),
            ("remaining_slivers", self.remaining_slivers),
            ("work_units", self.work_units),
            ("vertex_insertions", self.vertex_insertions),
            ("multiface_removals", self.multiface_removals),
        )


@final
class TetMeshExudeCounters(StrictModule):
    """Named tet mesh exude counters from the native ABI slots."""

    trials: int = eqx.field(static=True, default=0)
    flips: int = eqx.field(static=True, default=0)
    weighted_vertices: int = eqx.field(static=True, default=0)
    passes: int = eqx.field(static=True, default=0)
    remaining_slivers: int = eqx.field(static=True, default=0)
    work_units: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> TetMeshExudeCounters:
        if values.shape != (6,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "TetMeshExudeCounters.slots must have shape (6,) and dtype int64; got incompatible native slots"
            )
        return cls(
            trials=int(values[0]),
            flips=int(values[1]),
            weighted_vertices=int(values[2]),
            passes=int(values[3]),
            remaining_slivers=int(values[4]),
            work_units=int(values[5]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("trials", self.trials),
            ("flips", self.flips),
            ("weighted_vertices", self.weighted_vertices),
            ("passes", self.passes),
            ("remaining_slivers", self.remaining_slivers),
            ("work_units", self.work_units),
        )


@final
class Plc3DCounters(StrictModule):
    """Named plc3d counters from the native ABI slots."""

    rounds: int = eqx.field(static=True, default=0)
    input_triangles: int = eqx.field(static=True, default=0)
    facet_groups: int = eqx.field(static=True, default=0)
    plc_edges: int = eqx.field(static=True, default=0)
    protected_vertices: int = eqx.field(static=True, default=0)
    segment_steiner_points: int = eqx.field(static=True, default=0)
    facet_steiner_points: int = eqx.field(static=True, default=0)
    cavities: int = eqx.field(static=True, default=0)
    largest_cavity: int = eqx.field(static=True, default=0)
    gift_wrap_candidates: int = eqx.field(static=True, default=0)
    contact_tests: int = eqx.field(static=True, default=0)
    work_units: int = eqx.field(static=True, default=0)
    peak_retained_bytes: int = eqx.field(static=True, default=0)
    interior_steiner_points: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> Plc3DCounters:
        if values.shape != (14,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "Plc3DCounters.slots must have shape (14,) and dtype int64; got incompatible native slots"
            )
        return cls(
            rounds=int(values[0]),
            input_triangles=int(values[1]),
            facet_groups=int(values[2]),
            plc_edges=int(values[3]),
            protected_vertices=int(values[4]),
            segment_steiner_points=int(values[5]),
            facet_steiner_points=int(values[6]),
            cavities=int(values[7]),
            largest_cavity=int(values[8]),
            gift_wrap_candidates=int(values[9]),
            contact_tests=int(values[10]),
            work_units=int(values[11]),
            peak_retained_bytes=int(values[12]),
            interior_steiner_points=int(values[13]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("rounds", self.rounds),
            ("input_triangles", self.input_triangles),
            ("facet_groups", self.facet_groups),
            ("plc_edges", self.plc_edges),
            ("protected_vertices", self.protected_vertices),
            ("segment_steiner_points", self.segment_steiner_points),
            ("facet_steiner_points", self.facet_steiner_points),
            ("cavities", self.cavities),
            ("largest_cavity", self.largest_cavity),
            ("gift_wrap_candidates", self.gift_wrap_candidates),
            ("contact_tests", self.contact_tests),
            ("work_units", self.work_units),
            ("peak_retained_bytes", self.peak_retained_bytes),
            ("interior_steiner_points", self.interior_steiner_points),
        )


@final
class RestrictedPowerCounters(StrictModule):
    """Named power cells counters from the native ABI slots."""

    clips: int = eqx.field(static=True, default=0)
    pieces: int = eqx.field(static=True, default=0)
    empty_clips: int = eqx.field(static=True, default=0)
    walk_steps: int = eqx.field(static=True, default=0)
    merged_links: int = eqx.field(static=True, default=0)
    unmerged_fragment_groups: int = eqx.field(static=True, default=0)
    removed_collinear_vertices: int = eqx.field(static=True, default=0)
    cells: int = eqx.field(static=True, default=0)
    faces: int = eqx.field(static=True, default=0)
    vertices: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> RestrictedPowerCounters:
        if values.shape != (10,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "RestrictedPowerCounters.slots must have shape (10,) and dtype int64; got incompatible native slots"
            )
        return cls(
            clips=int(values[0]),
            pieces=int(values[1]),
            empty_clips=int(values[2]),
            walk_steps=int(values[3]),
            merged_links=int(values[4]),
            unmerged_fragment_groups=int(values[5]),
            removed_collinear_vertices=int(values[6]),
            cells=int(values[7]),
            faces=int(values[8]),
            vertices=int(values[9]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("clips", self.clips),
            ("pieces", self.pieces),
            ("empty_clips", self.empty_clips),
            ("walk_steps", self.walk_steps),
            ("merged_links", self.merged_links),
            ("unmerged_fragment_groups", self.unmerged_fragment_groups),
            ("removed_collinear_vertices", self.removed_collinear_vertices),
            ("cells", self.cells),
            ("faces", self.faces),
            ("vertices", self.vertices),
        )


@final
class ArrangementCounters(StrictModule):
    """Named arrangement counters from the native ABI slots."""

    pair_count: int = eqx.field(static=True, default=0)
    contact_pairs: int = eqx.field(static=True, default=0)
    coplanar_pairs: int = eqx.field(static=True, default=0)
    coincident_vertices: int = eqx.field(static=True, default=0)
    edge_plane_vertices: int = eqx.field(static=True, default=0)
    edge_edge_vertices: int = eqx.field(static=True, default=0)
    triple_plane_vertices: int = eqx.field(static=True, default=0)
    split_faces: int = eqx.field(static=True, default=0)
    fragments: int = eqx.field(static=True, default=0)
    filtered_predicates: int = eqx.field(static=True, default=0)
    exact_predicates: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> ArrangementCounters:
        if values.shape != (11,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "ArrangementCounters.slots must have shape (11,) and dtype int64; got incompatible native slots"
            )
        return cls(
            pair_count=int(values[0]),
            contact_pairs=int(values[1]),
            coplanar_pairs=int(values[2]),
            coincident_vertices=int(values[3]),
            edge_plane_vertices=int(values[4]),
            edge_edge_vertices=int(values[5]),
            triple_plane_vertices=int(values[6]),
            split_faces=int(values[7]),
            fragments=int(values[8]),
            filtered_predicates=int(values[9]),
            exact_predicates=int(values[10]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("pair_count", self.pair_count),
            ("contact_pairs", self.contact_pairs),
            ("coplanar_pairs", self.coplanar_pairs),
            ("coincident_vertices", self.coincident_vertices),
            ("edge_plane_vertices", self.edge_plane_vertices),
            ("edge_edge_vertices", self.edge_edge_vertices),
            ("triple_plane_vertices", self.triple_plane_vertices),
            ("split_faces", self.split_faces),
            ("fragments", self.fragments),
            ("filtered_predicates", self.filtered_predicates),
            ("exact_predicates", self.exact_predicates),
        )


@final
class RestrictedDualCounters(StrictModule):
    """Named restricted dual counters from the native ABI slots."""

    work_units: int = eqx.field(static=True, default=0)
    interior_facets: int = eqx.field(static=True, default=0)
    hull_facets: int = eqx.field(static=True, default=0)
    degenerate_facets: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> RestrictedDualCounters:
        if values.shape != (4,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "RestrictedDualCounters.slots must have shape (4,) and dtype int64; got incompatible native slots"
            )
        return cls(
            work_units=int(values[0]),
            interior_facets=int(values[1]),
            hull_facets=int(values[2]),
            degenerate_facets=int(values[3]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("work_units", self.work_units),
            ("interior_facets", self.interior_facets),
            ("hull_facets", self.hull_facets),
            ("degenerate_facets", self.degenerate_facets),
        )


@final
class PeriodicDelaunayEvidence(StrictModule):
    """Named periodic evidence from the native ABI slots."""

    rounds: float = eqx.field(static=True, default=0.0)
    margin: float = eqx.field(static=True, default=0.0)
    image_count: float = eqx.field(static=True, default=0.0)
    uncertified_cells: float = eqx.field(static=True, default=0.0)
    required_margin: float = eqx.field(static=True, default=0.0)
    finite_cell_slots: float = eqx.field(static=True, default=0.0)
    exact_evaluations: float = eqx.field(static=True, default=0.0)
    perturbed_decisions: float = eqx.field(static=True, default=0.0)
    exhausted_limit: float = eqx.field(static=True, default=0.0)
    duplicate_representative: float = eqx.field(static=True, default=0.0)

    @classmethod
    def from_slots(cls, values: NDArray[np.float64], /) -> PeriodicDelaunayEvidence:
        if values.shape != (10,) or values.dtype != np.dtype(np.float64):
            raise ValueError(
                "PeriodicDelaunayEvidence.slots must have shape (10,) and dtype float64; got incompatible native slots"
            )
        return cls(
            rounds=float(values[0]),
            margin=float(values[1]),
            image_count=float(values[2]),
            uncertified_cells=float(values[3]),
            required_margin=float(values[4]),
            finite_cell_slots=float(values[5]),
            exact_evaluations=float(values[6]),
            perturbed_decisions=float(values[7]),
            exhausted_limit=float(values[8]),
            duplicate_representative=float(values[9]),
        )

    def items(self) -> tuple[tuple[str, float], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("rounds", self.rounds),
            ("margin", self.margin),
            ("image_count", self.image_count),
            ("uncertified_cells", self.uncertified_cells),
            ("required_margin", self.required_margin),
            ("finite_cell_slots", self.finite_cell_slots),
            ("exact_evaluations", self.exact_evaluations),
            ("perturbed_decisions", self.perturbed_decisions),
            ("exhausted_limit", self.exhausted_limit),
            ("duplicate_representative", self.duplicate_representative),
        )


@final
class Triangulation3DStatistics(StrictModule):
    """Named triangulation 3d statistics from the native ABI slots."""

    vertex_ids: int = eqx.field(static=True, default=0)
    live_vertices: int = eqx.field(static=True, default=0)
    finite_cells: int = eqx.field(static=True, default=0)
    ghost_cells: int = eqx.field(static=True, default=0)
    cell_slots: int = eqx.field(static=True, default=0)
    free_slots: int = eqx.field(static=True, default=0)
    constrained_facets: int = eqx.field(static=True, default=0)
    work: int = eqx.field(static=True, default=0)
    committed_insertions: int = eqx.field(static=True, default=0)
    refused_insertions: int = eqx.field(static=True, default=0)
    largest_cavity: int = eqx.field(static=True, default=0)
    retained_bytes: int = eqx.field(static=True, default=0)
    peak_retained_bytes: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> Triangulation3DStatistics:
        if values.shape != (13,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "Triangulation3DStatistics.slots must have shape (13,) and dtype int64; got incompatible native slots"
            )
        return cls(
            vertex_ids=int(values[0]),
            live_vertices=int(values[1]),
            finite_cells=int(values[2]),
            ghost_cells=int(values[3]),
            cell_slots=int(values[4]),
            free_slots=int(values[5]),
            constrained_facets=int(values[6]),
            work=int(values[7]),
            committed_insertions=int(values[8]),
            refused_insertions=int(values[9]),
            largest_cavity=int(values[10]),
            retained_bytes=int(values[11]),
            peak_retained_bytes=int(values[12]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("vertex_ids", self.vertex_ids),
            ("live_vertices", self.live_vertices),
            ("finite_cells", self.finite_cells),
            ("ghost_cells", self.ghost_cells),
            ("cell_slots", self.cell_slots),
            ("free_slots", self.free_slots),
            ("constrained_facets", self.constrained_facets),
            ("work", self.work),
            ("committed_insertions", self.committed_insertions),
            ("refused_insertions", self.refused_insertions),
            ("largest_cavity", self.largest_cavity),
            ("retained_bytes", self.retained_bytes),
            ("peak_retained_bytes", self.peak_retained_bytes),
        )


@final
class SurfaceEnvelopeCounts(StrictModule):
    """Named surface envelope counts from the native ABI slots."""

    surface_vertices: int = eqx.field(static=True, default=0)
    surface_triangles: int = eqx.field(static=True, default=0)
    grid_samples: int = eqx.field(static=True, default=0)
    work_units: int = eqx.field(static=True, default=0)
    source_samples: int = eqx.field(static=True, default=0)
    volume_vertices: int = eqx.field(static=True, default=0)
    tetrahedra: int = eqx.field(static=True, default=0)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> SurfaceEnvelopeCounts:
        if values.shape != (7,) or values.dtype != np.dtype(np.int64):
            raise ValueError(
                "SurfaceEnvelopeCounts.slots must have shape (7,) and dtype int64; got incompatible native slots"
            )
        return cls(
            surface_vertices=int(values[0]),
            surface_triangles=int(values[1]),
            grid_samples=int(values[2]),
            work_units=int(values[3]),
            source_samples=int(values[4]),
            volume_vertices=int(values[5]),
            tetrahedra=int(values[6]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values for evidence and persistence payloads."""
        return (
            ("surface_vertices", self.surface_vertices),
            ("surface_triangles", self.surface_triangles),
            ("grid_samples", self.grid_samples),
            ("work_units", self.work_units),
            ("source_samples", self.source_samples),
            ("volume_vertices", self.volume_vertices),
            ("tetrahedra", self.tetrahedra),
        )


@final
class LevelSetCounters(StrictModule):
    """Measured native zero-insertion work and its failure edge."""

    vertices: int = eqx.field(static=True)
    cells: int = eqx.field(static=True)
    inserted: int = eqx.field(static=True)
    zero_vertices: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    failed_edge: int = eqx.field(static=True)

    @classmethod
    def from_slots(cls, values: NDArray[np.int64], /) -> LevelSetCounters:
        return cls(
            vertices=int(values[0]),
            cells=int(values[1]),
            inserted=int(values[2]),
            zero_vertices=int(values[3]),
            work_units=int(values[4]),
            failed_edge=int(values[5]),
        )

    def items(self) -> tuple[tuple[str, int], ...]:
        return (
            ("vertices", self.vertices),
            ("cells", self.cells),
            ("inserted", self.inserted),
            ("zero_vertices", self.zero_vertices),
            ("work_units", self.work_units),
            ("failed_edge", self.failed_edge),
        )


class LevelSetOutputMode(IntEnum):
    """Native level-set publication: validated carriers or topology only."""

    FULL_CARRIERS = 0
    TOPOLOGY_ONLY = 1


@final
class LevelSetSplitRequest(StrictModule):
    """Prepared zero-insertion inputs and caller-owned host output buffers.

    ``TOPOLOGY_ONLY`` publishes no carrier points or deviations, so both
    buffers are ``None``; ``FULL_CARRIERS`` requires both.
    """

    points: NDArray[np.float64]
    values: NDArray[np.float64]
    cells: NDArray[np.int32]
    faces: NDArray[np.int32]
    edges: NDArray[np.int32]
    protected_edges: NDArray[np.int8]
    max_vertices: int = eqx.field(static=True)
    max_cells: int = eqx.field(static=True)
    work_limit: int = eqx.field(static=True)
    cavity_limit: int = eqx.field(static=True)
    output_mode: LevelSetOutputMode = eqx.field(static=True)
    output_points: NDArray[np.float64] | None
    output_cells: NDArray[np.int32]
    output_parents: NDArray[np.int32]
    output_sources: NDArray[np.int32]
    output_weights: NDArray[np.float64]
    output_deviations: NDArray[np.float64] | None
    counter_slots: NDArray[np.int64]


@final
class LevelSetRun(StrictModule):
    """Native zero-insertion status, publication mode and measured work."""

    status: MeshcoreStatus = eqx.field(static=True)
    output_mode: LevelSetOutputMode = eqx.field(static=True)
    counters: LevelSetCounters


@checked
def level_set_split_3d(request: LevelSetSplitRequest, /) -> LevelSetRun:
    """Execute prepared zero insertion into caller-owned host buffers."""
    points, deviations = request.output_points, request.output_deviations
    match request.output_mode:
        case LevelSetOutputMode.FULL_CARRIERS:
            if points is None or deviations is None:
                raise ValueError(
                    "LevelSetSplitRequest.output_points must be a buffer together with output_deviations for FULL_CARRIERS; got a missing carrier buffer."
                )
            points_out = points.ctypes.data
            deviations_out = deviations.ctypes.data
        case LevelSetOutputMode.TOPOLOGY_ONLY:
            if points is not None or deviations is not None:
                raise ValueError(
                    "LevelSetSplitRequest.output_points must be None together with output_deviations for TOPOLOGY_ONLY; got a carrier buffer."
                )
            points_out = deviations_out = None
        case invalid:
            assert_never(invalid)
    status = load_meshcore()["phx_mc_level_set_split_3d"](
        ctypes.byref(
            _LevelSetSplit3DRequest(
                point_count=request.points.shape[0],
                points=request.points.ctypes.data,
                values=request.values.ctypes.data,
                cell_count=request.cells.shape[0],
                cells=request.cells.ctypes.data,
                face_count=request.faces.shape[0],
                faces=request.faces.ctypes.data,
                edge_count=request.edges.shape[0],
                edges=request.edges.ctypes.data,
                protected_edges=request.protected_edges.ctypes.data,
                max_vertices=request.max_vertices,
                max_cells=request.max_cells,
                work_limit=request.work_limit,
                cavity_limit=request.cavity_limit,
                output_mode=int(request.output_mode),
                out_points=points_out,
                out_cells=request.output_cells.ctypes.data,
                out_parents=request.output_parents.ctypes.data,
                out_sources=request.output_sources.ctypes.data,
                out_weights=request.output_weights.ctypes.data,
                out_deviations=deviations_out,
                counters=request.counter_slots.ctypes.data,
            )
        )
    )
    return LevelSetRun(
        MeshcoreStatus(status),
        request.output_mode,
        LevelSetCounters.from_slots(request.counter_slots),
    )


@final
class ContactIntersections(StrictModule):
    """Contact classifications and bounded host constructions."""

    classes: NDArray[np.int8]
    point_counts: NDArray[np.int32]
    points: NDArray[np.float64]
    point_bounds: NDArray[np.float64]
    point_features: NDArray[np.int8]
    status: NDArray[np.int32]


@final
class PolygonIntersectionSimplices(StrictModule):
    """Triangle partitions and moments of polygon intersections."""

    simplices: NDArray[np.float64]
    simplex_counts: NDArray[np.int32]
    area: NDArray[np.float64]
    first_moment: NDArray[np.float64]
    status: NDArray[np.int32]


@final
class TetrahedronIntersectionSimplices(StrictModule):
    """Tetrahedron partitions and moments of tetrahedron intersections."""

    simplices: NDArray[np.float64]
    simplex_counts: NDArray[np.int32]
    volume: NDArray[np.float64]
    first_moment: NDArray[np.float64]
    status: NDArray[np.int32]


@final
class BoxHalfplaneClip(StrictModule):
    """Counterclockwise clipped box loops, labels, moments and item status."""

    vertices: NDArray[np.float64]
    edge_labels: NDArray[np.int32]
    vertex_counts: NDArray[np.int32]
    area: NDArray[np.float64]
    first_moment: NDArray[np.float64]
    status: NDArray[np.int32]


@final
class BoxHalfspaceClip(StrictModule):
    """Outward clipped box faces, labels, moments and item status."""

    vertices: NDArray[np.float64]
    vertex_counts: NDArray[np.int32]
    face_offsets: NDArray[np.int32]
    face_labels: NDArray[np.int32]
    face_vertices: NDArray[np.int32]
    face_counts: NDArray[np.int32]
    volume: NDArray[np.float64]
    first_moment: NDArray[np.float64]
    status: NDArray[np.int32]


@final
class NativeMeshArrays(StrictModule):
    """Copied native triangulation arrays after releasing the native handle."""

    points: NDArray[np.float64]
    cells: NDArray[np.int32]
    vertex_map: NDArray[np.int32]
    cell_constraints: NDArray[np.int32]
    cell_regions: NDArray[np.int32]


@final
class ConstrainedDelaunay2D(StrictModule):
    """Constrained triangles with input identity, segment lineage and status."""

    points: NDArray[np.float64]
    triangles: NDArray[np.int32]
    vertex_map: NDArray[np.int32]
    segment_ids: NDArray[np.int32]
    status: MeshcoreStatus = eqx.field(static=True)


@final
class GraphPartition(StrictModule):
    """Partition assignments and measured native work."""

    parts: NDArray[np.int32]
    counters: GraphPartitionCounters


@final
class SurfaceReconnection(StrictModule):
    """Accepted surface triangulation, point identities and reconnection work."""

    triangles: NDArray[np.int32]
    constrained: NDArray[np.bool_]
    vertex_ids: NDArray[np.int32]
    item_status: NDArray[np.int32]
    counters: SurfaceReconnectionCounters


@final
class PeriodicDelaunay(StrictModule):
    """Periodic cells, lattice shifts and image-neighborhood evidence."""

    status: MeshcoreStatus = eqx.field(static=True)
    vertices: NDArray[np.int32]
    shifts: NDArray[np.int32]
    evidence: PeriodicDelaunayEvidence


@final
class PeriodicRefinementCounters(StrictModule):
    """Deterministic work and topology evidence from quotient frontal refinement."""

    work_units: int = eqx.field(static=True)
    candidates: int = eqx.field(static=True)
    cavity_evaluations: int = eqx.field(static=True)
    cavity_cells: int = eqx.field(static=True)
    commits: int = eqx.field(static=True)
    image_refusals: int = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    hard_issues: int = eqx.field(static=True)
    edges: int = eqx.field(static=True)
    stop_reason: int = eqx.field(static=True)


@final
class PeriodicRefinement(StrictModule):
    """Refined orbit representatives, quotient topology, and admission evidence."""

    status: MeshcoreStatus = eqx.field(static=True)
    points: NDArray[np.float64]
    vertices: NDArray[np.int32]
    shifts: NDArray[np.int32]
    counters: PeriodicRefinementCounters


@final
class SurfaceEnvelope(StrictModule):
    """Wrapped surface, measured bounds and enclosed volume triangulation."""

    surface_points: NDArray[np.float64]
    surface_faces: NDArray[np.int64]
    counts: SurfaceEnvelopeCounts
    bounds: NDArray[np.float64]
    volume_points: NDArray[np.float64]
    cells: NDArray[np.int64]


@final
class RestrictedDual3D(StrictModule):
    """Restricted dual facets, incident cells, endpoints and item evidence."""

    facets: NDArray[np.int32]
    incident: NDArray[np.int32]
    endpoints: NDArray[np.float64]
    kinds: NDArray[np.int32]
    item_status: NDArray[np.int32]
    counters: RestrictedDualCounters


@final
class NativeSourceComplex(StrictModule):
    """Validated native source-row banks and construction witnesses."""

    faces: NDArray[np.int32]
    face_group_rows: NDArray[np.int32]
    face_tolerances: NDArray[np.float64]
    segments: NDArray[np.int32]
    segment_group_rows: NDArray[np.int32]
    segment_tolerances: NDArray[np.float64]
    witness_strata: NDArray[np.int8]
    witness_entities: NDArray[np.int32]
    witness_parameters: NDArray[np.float64]
    points: NDArray[np.float64]


@final
class NativeExecutionWorkCounters(StrictModule):
    """Own the six execution work slots in their documented native ABI order."""

    construction_work: int = eqx.field(static=True)
    source_queries: int = eqx.field(static=True)
    peak_cavity_cells: int = eqx.field(static=True)
    work_refusals: int = eqx.field(static=True)
    query_refusals: int = eqx.field(static=True)
    cavity_refusals: int = eqx.field(static=True)

    def __init__(
        self,
        construction_work: int = 0,
        source_queries: int = 0,
        peak_cavity_cells: int = 0,
        work_refusals: int = 0,
        query_refusals: int = 0,
        cavity_refusals: int = 0,
    ) -> None:
        values = tuple(
            bounded_integer(value, name, lower=0, upper=_MAX_COUNTER_VALUE)
            for name, value in (
                ("construction_work", construction_work),
                ("source_queries", source_queries),
                ("peak_cavity_cells", peak_cavity_cells),
                ("work_refusals", work_refusals),
                ("query_refusals", query_refusals),
                ("cavity_refusals", cavity_refusals),
            )
        )
        (
            self.construction_work,
            self.source_queries,
            self.peak_cavity_cells,
            self.work_refusals,
            self.query_refusals,
            self.cavity_refusals,
        ) = values

    @classmethod
    def from_slots(cls, values: NDArray[np.uint64], /) -> NativeExecutionWorkCounters:
        """Decode the native transport once; retain no positional array."""
        if values.shape != (6,) or values.dtype != np.dtype(np.uint64):
            raise ValueError(
                "NativeExecutionWorkCounters.slots must have shape (6,) and dtype uint64; "
                f"got {_input_summary(values)}."
            )
        return cls(*(int(value) for value in values))

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values in native ABI slot order."""
        return (
            ("construction_work", self.construction_work),
            ("source_queries", self.source_queries),
            ("peak_cavity_cells", self.peak_cavity_cells),
            ("work_refusals", self.work_refusals),
            ("query_refusals", self.query_refusals),
            ("cavity_refusals", self.cavity_refusals),
        )

    def validate_restored(self) -> None:
        """Validate constructor-bypassing archive restoration without coercion."""
        for name, value in self.items():
            if type(value) is not int or not 0 <= value <= _MAX_COUNTER_VALUE:
                raise ValueError(
                    f"NativeExecutionWorkCounters.{name} must be a Python uint64-range integer; "
                    f"got {_input_summary(value)}."
                )


@final
class NativeExecutionMemoryCounters(StrictModule):
    """Own the six native allocator slots, excluding external host upper bounds."""

    limit_bytes: int = eqx.field(static=True)
    live_bytes: int = eqx.field(static=True)
    peak_bytes: int = eqx.field(static=True)
    requested_bytes: int = eqx.field(static=True)
    allocations: int = eqx.field(static=True)
    refusals: int = eqx.field(static=True)

    def __init__(
        self,
        limit_bytes: int = 0,
        live_bytes: int = 0,
        peak_bytes: int = 0,
        requested_bytes: int = 0,
        allocations: int = 0,
        refusals: int = 0,
    ) -> None:
        values = tuple(
            bounded_integer(value, name, lower=0, upper=_MAX_COUNTER_VALUE)
            for name, value in (
                ("limit_bytes", limit_bytes),
                ("live_bytes", live_bytes),
                ("peak_bytes", peak_bytes),
                ("requested_bytes", requested_bytes),
                ("allocations", allocations),
                ("refusals", refusals),
            )
        )
        (
            self.limit_bytes,
            self.live_bytes,
            self.peak_bytes,
            self.requested_bytes,
            self.allocations,
            self.refusals,
        ) = values

    @classmethod
    def from_slots(cls, values: NDArray[np.uint64], /) -> NativeExecutionMemoryCounters:
        """Decode the native transport once; retain no positional array."""
        if values.shape != (6,) or values.dtype != np.dtype(np.uint64):
            raise ValueError(
                "NativeExecutionMemoryCounters.slots must have shape (6,) and dtype uint64; "
                f"got {_input_summary(values)}."
            )
        return cls(*(int(value) for value in values))

    def items(self) -> tuple[tuple[str, int], ...]:
        """Return named values in native ABI slot order."""
        return (
            ("limit_bytes", self.limit_bytes),
            ("live_bytes", self.live_bytes),
            ("peak_bytes", self.peak_bytes),
            ("requested_bytes", self.requested_bytes),
            ("allocations", self.allocations),
            ("refusals", self.refusals),
        )

    def validate_restored(self) -> None:
        """Validate constructor-bypassing archive restoration without coercion."""
        for name, value in self.items():
            if type(value) is not int or not 0 <= value <= _MAX_COUNTER_VALUE:
                raise ValueError(
                    f"NativeExecutionMemoryCounters.{name} must be a Python uint64-range integer; "
                    f"got {_input_summary(value)}."
                )


@final
class NativeExecutionEvidence(StrictModule):
    """Measured CPU-native phase work and managed allocations, including refusal."""

    work_counters: NativeExecutionWorkCounters
    memory_counters: NativeExecutionMemoryCounters
    elapsed_seconds: float = eqx.field(static=True)
    status: MeshcoreStatus = eqx.field(static=True)
    externally_charged_work: int = eqx.field(static=True)
    externally_charged_geometry_queries: int = eqx.field(static=True)
    native_primitive_queries: int = eqx.field(static=True)
    host_storage_live_bytes_upper: int = eqx.field(static=True)
    host_storage_peak_bytes_upper: int = eqx.field(static=True)
    managed_admission_peak_bytes_upper: int | None = eqx.field(static=True)
    prior_elapsed_seconds: float = eqx.field(static=True, default=0.0)

    def __post_init__(self) -> None:
        self.validate_restored()
        if type(self.work_counters) is not NativeExecutionWorkCounters:
            raise TypeError(
                "NativeExecutionEvidence.work_counters must be NativeExecutionWorkCounters; "
                f"got {_input_summary(self.work_counters)}."
            )
        if type(self.memory_counters) is not NativeExecutionMemoryCounters:
            raise TypeError(
                "NativeExecutionEvidence.memory_counters must be NativeExecutionMemoryCounters; "
                f"got {_input_summary(self.memory_counters)}."
            )
        if (
            self.host_storage_live_bytes_upper < 0
            or self.host_storage_peak_bytes_upper < self.host_storage_live_bytes_upper
        ):
            raise ValueError(
                f"NativeExecutionEvidence.host_storage_bounds must be ordered and nonnegative; got {(self.host_storage_live_bytes_upper, self.host_storage_peak_bytes_upper)}."
            )
        if (
            not math.isfinite(self.prior_elapsed_seconds)
            or self.prior_elapsed_seconds < 0.0
        ):
            raise ValueError(
                f"NativeExecutionEvidence.prior_elapsed_seconds must be finite and nonnegative; got {self.prior_elapsed_seconds}."
            )

    def validate_restored(self) -> None:
        """Reject fabricated or out-of-range managed admission observations."""
        value = self.managed_admission_peak_bytes_upper
        if value is not None and (
            type(value) is not int or not 0 <= value <= _MAX_COUNTER_VALUE
        ):
            raise ValueError(
                "NativeExecutionEvidence.managed_admission_peak_bytes_upper must be "
                f"None or a Python uint64-range integer; got {_input_summary(value)}."
            )


@final
class NativeExecutionLimits(StrictModule):
    """Mechanical work, query, cavity, managed-storage and deadline bounds."""

    maximum_work_units: int = eqx.field(static=True)
    maximum_geometry_queries: int = eqx.field(static=True)
    maximum_cavity_cells: int = eqx.field(static=True)
    maximum_managed_bytes: int = eqx.field(static=True)
    maximum_seconds: float = eqx.field(static=True)

    def __init__(
        self,
        maximum_work_units: int,
        maximum_geometry_queries: int,
        maximum_cavity_cells: int,
        maximum_managed_bytes: int,
        maximum_seconds: float,
    ) -> None:
        if any(
            value is None
            for value in (
                maximum_work_units,
                maximum_geometry_queries,
                maximum_cavity_cells,
                maximum_managed_bytes,
            )
        ):
            raise TypeError(
                "NativeExecutionLimits.counters must be explicit integers; got None."
            )
        work = _unsigned_budget(maximum_work_units, "maximum_work_units")
        queries = _unsigned_budget(maximum_geometry_queries, "maximum_geometry_queries")
        cavity = _unsigned_budget(maximum_cavity_cells, "maximum_cavity_cells")
        managed = _unsigned_budget(maximum_managed_bytes, "maximum_managed_bytes")
        if isinstance(maximum_seconds, (bool, np.bool_)) or not isinstance(
            maximum_seconds, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                "NativeExecutionLimits.maximum_seconds must be a nonnegative number; got unsupported scalar."
            )
        seconds = float(maximum_seconds)
        if math.isnan(seconds) or seconds < 0.0:
            raise ValueError(
                f"NativeExecutionLimits.maximum_seconds must be nonnegative; got {seconds}."
            )
        self.maximum_work_units = work
        self.maximum_geometry_queries = queries
        self.maximum_cavity_cells = cavity
        self.maximum_managed_bytes = managed
        self.maximum_seconds = seconds


@final
class RemainingAllowance(StrictModule):
    """Unreserved resources of the live request; observation never renews limits."""

    remaining_work_units: int = eqx.field(static=True)
    remaining_geometry_queries: int = eqx.field(static=True)
    remaining_cavity_cells: int = eqx.field(static=True)
    remaining_managed_bytes: int = eqx.field(static=True)
    remaining_seconds: float = eqx.field(static=True)
    status: MeshcoreStatus = eqx.field(static=True)


class _RemainingValues(NamedTuple):
    """One fresh private native observation, not a validated public PyTree."""

    remaining_work_units: int
    remaining_geometry_queries: int
    remaining_cavity_cells: int
    remaining_managed_bytes: int
    remaining_seconds: float
    status: MeshcoreStatus


class _RemainingSnapshot(ctypes.Structure):
    """Private C output storage; each observation owns its snapshot."""

    work: int
    queries: int
    cavity: int
    managed_bytes: int
    seconds: float
    _fields_ = [
        ("work", ctypes.c_uint64),
        ("queries", ctypes.c_uint64),
        ("cavity", ctypes.c_uint64),
        ("managed_bytes", ctypes.c_uint64),
        ("seconds", ctypes.c_double),
    ]


def _free_host_array(
    library: MeshcoreLibrary,
    handle: int,
    observer: weakref.ReferenceType[NativeExecutionScope] | None = None,
    ticket: weakref.ReferenceType[NativeBatchTicket] | None = None,
    origin: weakref.ReferenceType[NativeExecutionScope] | None = None,
    /,
) -> None:
    # This callable's identity authenticates native-backed NumPy buffers.
    # Optional weak observation must neither replace that capability nor keep
    # the request or its scientific owners alive.
    scope = None if observer is None else observer()
    release = None
    try:
        if scope is not None:
            release = scope._managed_observation_begin_release(ticket, origin=origin)
    finally:
        library["phx_mc_execution_free_host_array"](handle)
        if scope is not None and release is not None:
            scope._managed_observation_end_release(release)


def is_native_host_buffer(buffer: ctypes.Array, /) -> bool:
    finalizer = vars(buffer).get("_native_execution_finalizer")
    match finalizer:
        case weakref.finalize():
            native = finalizer.peek()
            return native is not None and native[1] is _free_host_array
        case _:
            return False


class NativeScopeAttachment(Protocol):
    """Upper-layer request state released when its root execution scope ends."""

    def close(self) -> None: ...


class NativePhaseRecorder(Protocol):
    """Record one measured native phase and its optional storage bound."""

    def __call__(
        self, phase: str, seconds: float, peak_bytes: int | None, invocations: int, /
    ) -> None: ...


@final
class _NativeExecutionUsage(ctypes.Structure):
    """Aligned ticket counters shared with the native execution owner."""

    host_work: int
    host_queries: int
    host_cavity: int
    host_bytes: int
    host_peak_cavity: int
    host_peak_bytes: int
    native_work: int
    native_queries: int
    native_cavity: int
    native_bytes: int
    native_peak_cavity: int
    native_peak_bytes: int
    reserved_work: int
    reserved_queries: int
    reserved_cavity: int
    reserved_bytes: int
    children_work: int
    children_queries: int
    children_peak_cavity: int
    children_peak_bytes: int
    imported_work: int
    imported_queries: int
    children_native_work: int
    children_native_queries: int
    children_live_bytes: int
    retained_host_bytes: int
    combined_peak_cavity: int
    combined_peak_bytes: int

    _fields_ = [
        ("host_work", ctypes.c_uint64),
        ("host_queries", ctypes.c_uint64),
        ("host_cavity", ctypes.c_uint64),
        ("host_bytes", ctypes.c_uint64),
        ("host_peak_cavity", ctypes.c_uint64),
        ("host_peak_bytes", ctypes.c_uint64),
        ("native_work", ctypes.c_uint64),
        ("native_queries", ctypes.c_uint64),
        ("native_cavity", ctypes.c_uint64),
        ("native_bytes", ctypes.c_uint64),
        ("native_peak_cavity", ctypes.c_uint64),
        ("native_peak_bytes", ctypes.c_uint64),
        ("reserved_work", ctypes.c_uint64),
        ("reserved_queries", ctypes.c_uint64),
        ("reserved_cavity", ctypes.c_uint64),
        ("reserved_bytes", ctypes.c_uint64),
        ("children_work", ctypes.c_uint64),
        ("children_queries", ctypes.c_uint64),
        ("children_peak_cavity", ctypes.c_uint64),
        ("children_peak_bytes", ctypes.c_uint64),
        ("imported_work", ctypes.c_uint64),
        ("imported_queries", ctypes.c_uint64),
        ("children_native_work", ctypes.c_uint64),
        ("children_native_queries", ctypes.c_uint64),
        ("children_live_bytes", ctypes.c_uint64),
        ("retained_host_bytes", ctypes.c_uint64),
        ("combined_peak_cavity", ctypes.c_uint64),
        ("combined_peak_bytes", ctypes.c_uint64),
    ]


@final
class NativeParallelExecution(AbstractContextManager[None]):
    """Lower one meshing policy to affine native thread and deadline state.

    Logical chunks and their resource reservations belong to each native batch.
    This context supplies no additional meshing thread request or scientific data.
    Omitted parallelism inherits the active policy, or one thread without a
    parent. Nested policies retain the earliest parent deadline.
    """

    def __init__(
        self, parallelism: int | None = None, maximum_seconds: float = math.inf, /
    ) -> None:
        self._requested_parallelism = (
            None
            if parallelism is None
            else bounded_integer(
                parallelism,
                "NativeParallelExecution.parallelism",
                lower=1,
                upper=2**31 - 1,
            )
        )
        self.parallelism = (
            1 if self._requested_parallelism is None else self._requested_parallelism
        )
        if isinstance(maximum_seconds, (bool, np.bool_)):
            raise TypeError(
                f"NativeParallelExecution.maximum_seconds must be real and nonnegative; got {maximum_seconds!r}."
            )
        seconds = float(maximum_seconds)
        if math.isnan(seconds) or seconds < 0.0:
            raise ValueError(
                f"NativeParallelExecution.maximum_seconds must be nonnegative; got {maximum_seconds!r}."
            )
        self._maximum_seconds = seconds
        self._handle = ctypes.c_void_p()
        self._library: MeshcoreLibrary | None = None
        self._thread: int | None = None
        self._started: float | None = None
        self._token: Token[NativeParallelExecution | None] | None = None
        self._closed = False

    def __enter__(self) -> None:
        if self._thread is not None or self._closed:
            raise RuntimeError(
                "NativeParallelExecution.lifecycle must enter once; got active or ended context."
            )
        parent = _active_parallel_execution.get()
        if self._requested_parallelism is None:
            self.parallelism = 1 if parent is None else parent.parallelism
        seconds = self._maximum_seconds
        if parent is not None:
            seconds = min(seconds, parent.remaining_seconds())
        library = load_meshcore()
        started = time.monotonic()
        _raise_for_call(
            library["phx_mc_parallel_begin"](
                self.parallelism, seconds, ctypes.addressof(self._handle)
            ),
            "parallel_begin",
        )
        self._maximum_seconds = seconds
        self._library = library
        self._started = started
        self._thread = threading.get_ident()
        self._token = _active_parallel_execution.set(self)
        return None

    def remaining_seconds(self) -> float:
        """Return the nonrenewing deadline allowance on its creating thread."""
        if self._thread != threading.get_ident() or self._started is None:
            raise RuntimeError(
                f"NativeParallelExecution.thread must be its active creating thread; got owner_thread={self._thread}, caller_thread={threading.get_ident()}."
            )
        return max(0.0, self._maximum_seconds - (time.monotonic() - self._started))

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if (
            self._thread != threading.get_ident()
            or self._library is None
            or self._token is None
            or _active_parallel_execution.get() is not self
        ):
            raise RuntimeError(
                "NativeParallelExecution.lifecycle must end its active creating-thread context in order; got inactive, foreign or nested context."
            )
        status = MeshcoreStatus(self._library["phx_mc_parallel_end"](self._handle))
        if status not in (MeshcoreStatus.OK, MeshcoreStatus.TIMEOUT):
            _raise_for_call(status, "parallel_end")
        _active_parallel_execution.reset(self._token)
        self._token = None
        self._thread = None
        self._closed = True
        self._handle = ctypes.c_void_p()
        if status is MeshcoreStatus.TIMEOUT and exception is None:
            raise MeshcoreError(
                status,
                "NativeParallelExecution.deadline must admit execution and publication; got expired policy deadline.",
                elapsed_seconds=(
                    None if self._started is None else time.monotonic() - self._started
                ),
                maximum_seconds=self._maximum_seconds,
            )


_active_parallel_execution: ContextVar[NativeParallelExecution | None] = ContextVar(
    "native_parallel_execution", default=None
)


class NativeExecutionScope:
    """One input CPU-native allowance, shared by nested FFI owners.

    This is not a launch or sharding runtime. The steady-clock deadline remains
    native throughout a monolithic call. ``mesh`` borrows an existing managed
    pool; scratch bounds additional live bytes above that pool's retained
    baseline. New construction borrows the active parent pool automatically.
    The scope's evidence survives success, resource refusal and Python errors.
    Context exit closes entered descendants and releases remaining reservations
    before ending the native allowance.
    """

    @checked
    def __init__(
        self,
        limits: NativeExecutionLimits,
        /,
        *,
        mesh: TetMesh3D | None = None,
    ) -> None:
        self._limits_record = limits
        self._limits = (
            limits.maximum_work_units,
            limits.maximum_geometry_queries,
            limits.maximum_cavity_cells,
            limits.maximum_managed_bytes,
        )
        self._seconds = limits.maximum_seconds
        self._effective_seconds = limits.maximum_seconds
        self._mesh = mesh
        self._handle = ctypes.c_void_p()
        self._thread: int | None = None
        self._library: MeshcoreLibrary | None = None
        self.evidence: NativeExecutionEvidence | None = None
        self._token: Token[NativeExecutionScope | None] | None = None
        self._parent: NativeExecutionScope | None = None
        self._borrowed_ticket: NativeBatchTicket | None = None
        self._target_stacks: dict[int, list[NativeBatchTicket]] = {}
        self._tickets: dict[NativeBatchTicket, None] = {}
        self._ticket_views: dict[NativeScopeTicket, None] = {}
        self._closed_by_owner = False
        self._operation_started: float | None = None
        self._surfaced_child_statuses: set[MeshcoreStatus] = set()
        self._preparation_attachment: NativeScopeAttachment | None = None
        self._attachment_lock = threading.RLock()
        self._managed_observer_owner: NativeExecutionScope | None = None
        self._managed_observation_parent: NativeExecutionScope | None = None
        self._managed_observation_capacity = limits.maximum_managed_bytes
        self._managed_observation_enabled = False
        self._managed_observation_complete = True
        self._managed_observation_peak = 0
        self._managed_admission_started = False
        self._managed_observation_roots: dict[NativeBatchTicket, int] = {}
        self._managed_observation_covered = False
        self._managed_observation_window: NativeExecutionScope | None = None
        self._managed_observation_previous_window: NativeExecutionScope | None = None
        self._managed_observation_window_active = False
        self._managed_observation_releasing = False
        self._managed_native_events_complete = False
        self._managed_native_event_busy = False
        self._managed_retained_native_depth = 0
        self._source_metadata: list[
            weakref.ReferenceType[NativeHostStorageWorkspace]
        ] = []
        self._source_metadata_completed = False

    def observe_managed_admission(self) -> None:
        """Opt in at lexical route entry, before admitting request-owned storage.

        Evidence is a fixed-quota admission envelope, not physical RSS or a sum
        of independent allocator/host peaks. Unknown native entries or outside
        growth retain ``None``. Repeated opt-in never resets incompleteness.

        An explicitly observed child owns a lexical tracker, not another
        allowance. Its entry baseline includes live outside storage, excludes
        unused ancestor escrow, and never imports an ancestor's historical peak.
        Completed child coverage composes into its parent by maximum; native
        counters, deadlines, targets, and refusal status retain their owners.
        The published admission-demand upper also covers native host settlement's
        conservative historical-phase upper; it is not a physical-memory peak.
        """
        library = self._active_native_library(
            "NativeExecutionScope.observe_managed_admission"
        )
        if self._managed_observation_enabled:
            return
        if self._parent is not None:
            previous = self._managed_observer_owner
            if self._managed_observation_window_active:
                assert previous is not None
                previous._managed_observation_window = (
                    self._managed_observation_previous_window
                )
                self._managed_observation_window_active = False
            self._managed_observation_parent = previous
            if previous is not None and (
                previous._managed_native_event_busy
                or (
                    self._borrowed_ticket is not None
                    and not previous._managed_observation_enabled
                )
            ):
                self._managed_observation_complete = False
            self._managed_observer_owner = None
        self._managed_native_events_complete = all(
            isinstance(function, CFuncPtr)
            and getattr(function, "errcheck", None) is _observe_managed_native_call
            for function in library._functions.values()
        )
        self._managed_observation_enabled = True
        if self._mesh is not None or self._managed_admission_started:
            self._managed_observation_complete = False
        self._managed_observation_peak = self._managed_observation_used()

    def _managed_observation_active_scope(self) -> NativeExecutionScope | None:
        owner = self._managed_observer_owner or self
        active = current_native_execution_scope()
        if (
            active is None
            or (active._managed_observer_owner or active) is not owner
            or (active is not owner and not active._managed_observation_covered)
        ):
            owner._managed_observation_complete = False
            return None
        return active

    def _managed_request_root(self) -> NativeExecutionScope:
        """Use one allocator-pool coordinate system across inherited targets."""
        root = self
        while root._parent is not None:
            root = root._parent
        return root

    def _managed_observation_used(self) -> int:
        owner = self._managed_observer_owner or self
        active = owner._managed_observation_active_scope()
        if active is None:
            return 0  # Incomplete observation is never published as this value.
        request = active._managed_request_root()
        used = (
            active._managed_observation_capacity
            - request._remaining_values().remaining_managed_bytes
        )
        borrowed = owner._borrowed_ticket
        if borrowed is not None:
            while borrowed.parent is not None:
                borrowed = borrowed.parent
            # An ancestor escrow is not a child allocation. Its live usage and
            # child reservations, unlike its historical peak, belong to this
            # lexical interval.
            used -= borrowed.remaining_managed_bytes
        return used

    def _managed_observation_target(
        self,
    ) -> weakref.ReferenceType[NativeBatchTicket] | None:
        ticket = self._target_for_thread()
        if ticket is None:
            ticket = self._borrowed_ticket
        owner = self._managed_observer_owner or self
        while (
            ticket is not None
            and ticket.parent is not None
            and (ticket.parent.scope._managed_observer_owner or ticket.parent.scope)
            is owner
        ):
            ticket = ticket.parent
        return None if ticket is None else weakref.ref(ticket)

    def _managed_observation_mutation(
        self,
        ticket: NativeBatchTicket | weakref.ReferenceType[NativeBatchTicket] | None,
        /,
        *,
        origin: weakref.ReferenceType[NativeExecutionScope] | None = None,
    ) -> None:
        owner = self._managed_observer_owner or self
        owner._managed_admission_started = True
        self._managed_admission_started = True
        if not owner._managed_observation_enabled or owner._thread is None:
            return
        if isinstance(ticket, weakref.ReferenceType):
            ticket = ticket()
        source = (
            origin() if origin is not None else (self if ticket is None else ticket.scope)
        )
        window = owner._managed_observation_window
        if window is not None:
            while source is not None and source is not window:
                source = source._parent
            if source is None:
                # A nonbinding child window is only a root bottleneck while
                # all reservations outside its entry baseline stay held.
                owner._managed_observation_complete = False
        for root in tuple(owner._managed_observation_roots):
            if root is not ticket:
                # Its saved outside baseline is no longer held throughout the
                # local peak's interval. Do not invent synchronized history.
                owner._managed_observation_complete = False

    @contextmanager
    def _managed_retained_native_call(self) -> Iterator[None]:
        """Authorize only an elementary retained allocator's live-owner hook."""
        owner = self._managed_observer_owner or self
        owner._managed_retained_native_depth += 1
        try:
            yield
        finally:
            owner._managed_retained_native_depth -= 1

    def _managed_native_entry(self, name: str, function: object, /) -> None:
        owner = self._managed_observer_owner or self
        if name not in _MANAGED_NATIVE_CONTROL and name not in _MANAGED_NATIVE_FREE:
            self._managed_admission_started = True
        if not owner._managed_observation_enabled:
            if name not in _MANAGED_NATIVE_CONTROL and name not in _MANAGED_NATIVE_FREE:
                owner._managed_admission_started = True
            return
        if (
            self._library is None
            or self._library._functions.get(name) is not function
            or owner._managed_native_event_busy
            or owner._thread != threading.get_ident()
        ):
            owner._managed_native_events_complete = False
            return
        if name in _MANAGED_NATIVE_CONTROL or name in _MANAGED_NATIVE_FREE:
            return
        if name in _MANAGED_NATIVE_RETAINED:
            if owner._managed_retained_native_depth == 0:
                owner._managed_native_events_complete = False
            return
        owner._managed_native_event_busy = True
        try:
            target = self._target_for_thread()
            if target is None:
                target = self._borrowed_target_for_thread()
            if target is None or not target._mapped:
                owner._managed_native_events_complete = False
        finally:
            owner._managed_native_event_busy = False

    def _managed_observation_begin_release(
        self,
        ticket: NativeBatchTicket | weakref.ReferenceType[NativeBatchTicket] | None,
        /,
        *,
        origin: weakref.ReferenceType[NativeExecutionScope] | None = None,
    ) -> tuple[int, tuple[tuple[NativeBatchTicket, int], ...]] | None:
        """Checkpoint a phase before a real outside reservation decreases."""
        owner = self._managed_observer_owner or self
        if (
            not owner._managed_observation_enabled
            or not owner._managed_observation_complete
            or owner._thread is None
        ):
            return None
        if owner._thread != threading.get_ident():
            owner._managed_observation_complete = False
            return None
        if owner._managed_observation_releasing:
            owner._managed_observation_complete = False
            return None
        if isinstance(ticket, weakref.ReferenceType):
            ticket = ticket()
        source = (
            origin() if origin is not None else (self if ticket is None else ticket.scope)
        )
        window = owner._managed_observation_window
        if window is not None:
            while source is not None and source is not window:
                source = source._parent
            if source is None:
                owner._managed_observation_complete = False
                return None
        owner._managed_observation_releasing = True
        outside = []
        for root, baseline in tuple(owner._managed_observation_roots.items()):
            ancestor = ticket
            while ancestor is not None and ancestor is not root:
                ancestor = ancestor.parent
            if ancestor is root:
                # This token belongs inside the phase, not its outside bank.
                continue
            root._observe_capacity()
            owner._managed_observation_peak = max(
                owner._managed_observation_peak,
                baseline + max(root._peak_bytes, root._managed_admission_peak),
            )
            outside.append((root, baseline))
        return owner._managed_observation_used(), tuple(outside)

    def _managed_observation_end_release(
        self, release: tuple[int, tuple[tuple[NativeBatchTicket, int], ...]], /
    ) -> None:
        owner = self._managed_observer_owner or self
        try:
            if not owner._managed_observation_complete or owner._thread is None:
                return
            before, outside = release
            released = before - owner._managed_observation_used()
            if released < 0:
                owner._managed_observation_complete = False
                return
            for root, baseline in outside:
                if (
                    root not in owner._managed_observation_roots
                    or owner._managed_observation_roots[root] != baseline
                    or released > baseline
                ):
                    owner._managed_observation_complete = False
                    return
                # The phase's own quota was fixed at entry. Releasing outside
                # storage cannot enlarge it: the original entry baseline plus
                # a later phase peak remains the genuine admission requirement,
                # even when the corresponding physical peaks are disjoint.
            owner._managed_observation_candidate()
        finally:
            owner._managed_observation_releasing = False

    def _managed_observation_candidate(self) -> None:
        owner = self._managed_observer_owner or self
        if (
            owner._managed_observation_enabled
            and owner._managed_observation_complete
            and not owner._managed_observation_roots
            and owner._thread is not None
        ):
            if owner._thread != threading.get_ident():
                # Worker activity without a covering root phase has no affine
                # simultaneous outside snapshot.
                owner._managed_observation_complete = False
                return
            # Cache tickets remain genuine outside reservations during a work
            # phase. At a direct checkpoint, however, their unused escrow is
            # not simultaneous usage and must not echo the request's cap.
            active = owner._managed_observation_active_scope()
            if active is None:
                return
            unused = 0
            while active is not None:
                unused += sum(
                    ticket.remaining_managed_bytes
                    for ticket in tuple(active._tickets)
                    if ticket.parent is None
                )
                if active is owner:
                    break
                active = active._parent
            candidate = owner._managed_observation_used() - unused
            if candidate < 0:
                owner._managed_observation_complete = False
                return
            owner._managed_observation_peak = max(
                owner._managed_observation_peak, candidate
            )

    def _managed_observation_admit(self, ticket: NativeBatchTicket, /) -> None:
        owner = self._managed_observer_owner or self
        if not (ticket._managed_observation_phase or ticket.bytes_upper):
            # A zero-resource deadline checkpoint neither reserves storage nor
            # begins a phase whose history an opt-in observer could miss.
            return
        owner._managed_admission_started = True
        self._managed_admission_started = True
        if (
            owner._managed_observation_enabled
            and not owner._managed_observation_roots
            and owner._thread != threading.get_ident()
        ):
            owner._managed_observation_complete = False
        if not owner._managed_observation_enabled or (
            ticket.parent is not None
            and (ticket.parent.scope._managed_observer_owner or ticket.parent.scope)
            is owner
        ):
            return
        owner._managed_observation_mutation(ticket)
        if ticket.scope is not owner and not ticket.scope._managed_observation_covered:
            owner._managed_observation_complete = False
            return
        if not ticket._managed_observation_phase:
            owner._managed_observation_candidate()
            return
        outside = owner._managed_observation_used() - ticket.bytes_upper
        if outside < 0:
            owner._managed_observation_complete = False
            return
        owner._managed_observation_roots[ticket] = outside

    def _managed_observation_activate(self, ticket: NativeBatchTicket, /) -> None:
        owner = self._managed_observer_owner or self
        if (
            (
                ticket.parent is not None
                and (ticket.parent.scope._managed_observer_owner or ticket.parent.scope)
                is owner
            )
            or ticket._managed_observation_phase
            or ticket.bytes_upper == 0
        ):
            return
        ticket._managed_observation_phase = True
        if ticket._settled:
            return
        if ticket._peak_bytes:
            # A cache converted after prior usage has no synchronized outside
            # history for that earlier local peak.
            owner._managed_observation_complete = False
        self._managed_observation_admit(ticket)

    def _managed_observation_settle(self, ticket: NativeBatchTicket, /) -> None:
        owner = self._managed_observer_owner or self
        if not (ticket._managed_observation_phase or ticket.bytes_upper):
            return
        if not owner._managed_observation_enabled or (
            ticket.parent is not None
            and (ticket.parent.scope._managed_observer_owner or ticket.parent.scope)
            is owner
        ):
            return
        if not ticket._managed_observation_phase:
            owner._managed_observation_mutation(ticket)
            owner._managed_observation_candidate()
            return
        outside = owner._managed_observation_roots.pop(ticket, None)
        if outside is None:
            owner._managed_observation_complete = False
            return
        owner._managed_observation_peak = max(
            owner._managed_observation_peak,
            outside + max(ticket._peak_bytes, ticket._managed_admission_peak),
        )
        owner._managed_observation_mutation(ticket)

    @staticmethod
    def _managed_observation_release_host_storage(
        observer: weakref.ReferenceType[NativeExecutionScope],
        library: MeshcoreLibrary,
        handle: int,
        ticket: weakref.ReferenceType[NativeBatchTicket] | None,
        origin: weakref.ReferenceType[NativeExecutionScope] | None,
        /,
    ) -> None:
        scope = observer()
        release = None
        try:
            if scope is not None:
                release = scope._managed_observation_begin_release(ticket, origin=origin)
        finally:
            library["phx_mc_execution_release_host_storage"](handle)
            if scope is not None and release is not None:
                scope._managed_observation_end_release(release)

    def _managed_observation_result(
        self, native_peak: int, host_peak: int, /
    ) -> int | None:
        if (
            not self._managed_observation_enabled
            or not self._managed_observation_complete
            or self._managed_observation_releasing
            or self._managed_observation_roots
            # Every allocation-bearing typed entry must be associated with a
            # real mapped phase, or an elementary retained-allocation hook.
            # Independent native and host maxima are not a simultaneity proof.
            or not self._managed_native_events_complete
            or self._managed_observation_peak < native_peak
        ):
            return None
        request_limit = self._managed_request_root()._limits_record.maximum_managed_bytes
        if self._managed_observation_peak > request_limit:
            # A complete event envelope cannot exceed the enforced request pool.
            return None
        # Settlement's historical host upper may combine disjoint intervals.
        # The enforced request quota independently bounds admitted managed demand;
        # intersect only that conservative component, never a measured event.
        return max(self._managed_observation_peak, min(host_peak, request_limit))

    def completed_evidence(self) -> NativeExecutionEvidence:
        """Return the completed scope's measured work and storage evidence."""
        if self.evidence is None:
            raise RuntimeError(
                "NativeExecutionScope.evidence must describe an ended execution; got no completed evidence."
            )
        return self.evidence

    def preparation_ledger(
        self, factory: Callable[[], NativeScopeAttachment], /
    ) -> NativeScopeAttachment:
        """Return the request's single upper-layer preparation attachment."""
        owner = self
        while owner._parent is not None:
            owner = owner._parent
        if owner._thread is None:
            raise RuntimeError(
                "NativeExecutionScope.preparation_ledger must belong to a live request; got inactive scope."
            )
        with owner._attachment_lock:
            if owner._preparation_attachment is None:
                owner._preparation_attachment = factory()
            return owner._preparation_attachment

    def _target_for_thread(self) -> NativeBatchTicket | None:
        stack = self._target_stacks.get(threading.get_ident())
        return None if not stack else stack[-1]

    def _borrowed_target_for_thread(self) -> NativeBatchTicket | None:
        """Resolve only the still-mapped, same-thread target of a live ancestor."""
        ticket = self._borrowed_ticket
        if ticket is None:
            return None
        ancestor = self._parent
        while ancestor is not None and ancestor is not ticket.scope:
            ancestor = ancestor._parent
        if (
            ancestor is None
            or ticket.settled
            or not ticket._mapped
            or ticket._usage_thread != threading.get_ident()
            or ticket.scope._target_for_thread() is not ticket
            or ticket._target is None
            or ticket._target._closed
        ):
            return None
        return ticket

    def _native_thread_active(self) -> bool:
        return (
            self._thread == threading.get_ident() or self._target_for_thread() is not None
        )

    def _active_native_library(self, operation: str, /) -> MeshcoreLibrary:
        if not self._native_thread_active() or self._library is None:
            raise RuntimeError(
                f"{operation + '.scope'} must use its creating thread or explicit native target; "
                f"got owner_thread={self._thread}, caller_thread={threading.get_ident()}, "
                f"library_loaded={self._library is not None}."
            )
        return self._library

    def allocator_bookkeeping(self) -> int:
        """Observe retained allocator metadata separately from managed payload bytes."""
        if self._thread is None or self._library is None:
            raise RuntimeError(
                "NativeExecutionScope.allocator_bookkeeping must use a live scope; got inactive scope."
            )
        values = np.zeros((1,), dtype=np.uint64)
        _raise_for_call(
            self._library["phx_mc_execution_allocator_bookkeeping"](
                self._handle, values.ctypes.data
            ),
            "execution_allocator_bookkeeping",
        )
        return int(values[0])

    def batch(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NativeBatchTicket:
        """Reserve a batch, borrowing the corresponding Python target parent."""
        parent = self._target_for_thread()
        if parent is None:
            parent = self._borrowed_ticket
        return NativeBatchTicket(
            self, parent, work_upper, queries_upper, cavity_upper, bytes_upper
        )

    def ticket_view(self) -> NativeScopeTicket:
        """View this actual allowance without reserving native-owner headroom."""
        self._active_native_library("NativeExecutionScope.ticket_view")
        return NativeScopeTicket(self)

    def admit_source_metadata(
        self, bytes_upper: int, work_units: int, /
    ) -> NativeHostStorageWorkspace:
        """Prepay a metadata-only publication lifetime on this live request root.

        The returned existing storage owner may be cancelled with ``close``.
        After successful root completion it may hand its unchanged token to a
        weak final source; no payload graph or request workspace is retained.
        """
        if current_native_execution_scope() is not self or self._parent is not None:
            raise RuntimeError(
                "Source metadata admission requires its live request root."
            )
        self._active_native_library("NativeExecutionScope.admit_source_metadata")
        size = _unsigned_budget(bytes_upper, "bytes_upper")
        work = _unsigned_budget(work_units, "work_units")
        storage = NativeHostStorageWorkspace(self).open()
        try:
            storage.set_bound(size)
            storage.reserve_empty()
            self.charge(work=work)
            self._source_metadata.append(weakref.ref(storage))
            storage._source_metadata = True
        except BaseException:
            storage.close()
            raise
        return storage

    @property
    def limits(self) -> NativeExecutionLimits:
        """Return the mechanical limits record supplied at construction."""
        return self._limits_record

    @property
    def operation_started(self) -> float:
        """Return the request's monotonic start, debited by imported prior time."""
        owner = self
        while owner._parent is not None:
            owner = owner._parent
        started = owner._operation_started
        if started is None:
            raise RuntimeError(
                "NativeExecutionScope.operation_started must follow scope entry; got an unstarted request."
            )
        return started

    @property
    def parent(self) -> NativeExecutionScope | None:
        """Return the parent saved at entry, including after this phase ends."""
        return self._parent

    def __enter__(self) -> NativeExecutionScope:
        if self._thread is not None or self.evidence is not None:
            raise RuntimeError(
                f"_meshcore.NativeExecutionScope.__enter__.self must satisfy the input contract (A native execution budget is single-use.); got {_input_summary(self)}."
            )
        self._parent = current_native_execution_scope()
        request_root = self._managed_request_root()
        parent_remaining = None
        if self._parent is not None:
            parent_remaining = request_root.remaining().remaining_managed_bytes
            self._managed_observation_capacity = (
                self._parent._managed_observation_capacity
            )
        if self._parent is not None:
            self._borrowed_ticket = self._parent._target_for_thread()
            if self._borrowed_ticket is None:
                self._borrowed_ticket = self._parent._borrowed_ticket
            self._managed_observer_owner = (
                self._parent._managed_observer_owner or self._parent
            )
            owner = self._managed_observer_owner
            if owner._managed_observation_enabled:
                if self._mesh is not None:
                    owner._managed_observation_complete = False
                elif self._borrowed_ticket is not None:
                    # Stop at this observer's admitted root, not an independently
                    # observed ancestor's escrow above it.
                    root = self._managed_observation_target()
                    self._managed_observation_covered = (
                        root is not None and root() in owner._managed_observation_roots
                    )
                elif (
                    (self._parent is owner or self._parent._managed_observation_covered)
                    and self._limits_record.maximum_managed_bytes
                    >= self._parent.remaining().remaining_managed_bytes
                ):
                    # The child cannot become the tighter managed window while
                    # its outside entry baseline remains held.
                    self._managed_observation_covered = True
                    self._managed_observation_window_active = True
        operation_started = (
            time.monotonic() if self._parent is None else self._parent.operation_started
        )
        library = load_meshcore()
        seconds = self._seconds
        effective_seconds = seconds
        parallel = _active_parallel_execution.get()
        if parallel is not None:
            if parallel._library is not library:
                raise RuntimeError(
                    "NativeExecutionScope.library must match its lowered parallel policy; got another library."
                )
            effective_seconds = min(effective_seconds, parallel.remaining_seconds())
        if self._parent is not None:
            effective_seconds = min(
                effective_seconds, self._parent.remaining().remaining_seconds
            )
        # Native keeps the original request clock separate from the absolute
        # parallel-policy expiry and ancestor clocks. Pre-clamping this input
        # would debit already elapsed preparation from the workflow remainder.
        _raise_for_call(
            library["phx_mc_execution_begin"](
                *self._limits,
                seconds,
                None if self._mesh is None else self._mesh._live(),
                ctypes.addressof(self._handle),
            ),
            "execution_begin",
        )
        self._library = library
        self._effective_seconds = effective_seconds
        self._operation_started = operation_started
        self._thread = threading.get_ident()
        self._token = _active_execution_scope.set(self)
        _native_context_stack.set((*_native_context_stack.get(), self))
        if parent_remaining is not None:
            self._managed_observation_capacity -= (
                parent_remaining - request_root.remaining().remaining_managed_bytes
            )
        if self._managed_observation_window_active:
            owner = self._managed_observer_owner
            assert owner is not None
            self._managed_observation_previous_window = owner._managed_observation_window
            owner._managed_observation_window = self
        return self

    def charge(self, *, work: int = 0, geometry_queries: int = 0) -> None:
        """Admit separately owned host/device batch work before dispatch.

        Do not repeat native owner work or predicates already charged by the
        scope. This single batch barrier is not a per-candidate host callback.
        """
        if work is None or geometry_queries is None:
            raise TypeError(
                f"_meshcore.NativeExecutionScope.charge.work must be explicit integers; got {_input_summary(work)}."
            )
        library = self._active_native_library("NativeExecutionScope.charge")
        _raise_for_call(
            library["phx_mc_execution_charge"](
                self._handle,
                _unsigned_budget(work, "work"),
                _unsigned_budget(geometry_queries, "geometry_queries"),
            ),
            "execution_charge",
        )

    def import_preparation(
        self,
        *,
        work: int,
        geometry_queries: int,
        elapsed_seconds: float,
    ) -> None:
        """Debit a real ended predecessor, preserving raw scope elapsed."""
        if (
            isinstance(elapsed_seconds, (bool, np.bool_))
            or not isinstance(elapsed_seconds, (int, float, np.integer, np.floating))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0.0
        ):
            raise ValueError(
                f"NativeExecutionScope.elapsed_seconds must be finite and nonnegative; got {elapsed_seconds!r}."
            )
        library = self._active_native_library("NativeExecutionScope.import_preparation")
        _raise_for_call(
            library["phx_mc_execution_import_preparation"](
                self._handle,
                _unsigned_budget(work, "work"),
                _unsigned_budget(geometry_queries, "geometry_queries"),
                float(elapsed_seconds),
            ),
            "execution_import_preparation",
        )
        owner = self
        while owner._parent is not None:
            owner = owner._parent
        owner._operation_started = owner.operation_started - float(elapsed_seconds)

    def admit_work_bound(self, maximum_work: int, /) -> None:
        """Admit a pure-batch work bound without recording consumption.

        Charge measured work after execution and before committing candidates.
        Native operations continue to charge their own work.
        """
        if maximum_work is None:
            raise TypeError(
                f"_meshcore.NativeExecutionScope.admit_work_bound.maximum_work must be an explicit integer; got {_input_summary(maximum_work)}."
            )
        library = self._active_native_library("NativeExecutionScope.admit_work_bound")
        _raise_for_call(
            library["phx_mc_execution_admit_work_bound"](
                self._handle,
                _unsigned_budget(maximum_work, "maximum_work"),
            ),
            "execution_admit_work_bound",
        )

    def remaining(self) -> RemainingAllowance:
        """Observe parent allowance minima and available managed bytes."""
        return RemainingAllowance(*self._remaining_values())

    def _remaining_values(self) -> _RemainingValues:
        """Read every native counter and clock afresh without a public-module copy."""
        library = self._active_native_library("NativeExecutionScope.remaining")
        snapshot = _RemainingSnapshot()
        address = ctypes.addressof(snapshot)
        status = MeshcoreStatus(
            library["phx_mc_execution_remaining"](
                self._handle,
                address,
                address + 4 * ctypes.sizeof(ctypes.c_uint64),
            )
        )
        if status == MeshcoreStatus.INVALID_ARGUMENT:
            _raise_for_call(status, "execution_remaining")
        return _RemainingValues(
            snapshot.work,
            snapshot.queries,
            snapshot.cavity,
            snapshot.managed_bytes,
            snapshot.seconds,
            status,
        )

    def admit_cavity(self, count: int, /) -> None:
        """Admit a host edit's touched-cell set against the shared ledger."""
        library = self._active_native_library("NativeExecutionScope.admit_cavity")
        _raise_for_call(
            library["phx_mc_execution_admit_cavity"](
                self._handle,
                _unsigned_budget(count, "cavity_cells"),
            ),
            "execution_admit_cavity",
        )

    def admit_host_cavity(self, count: int, /) -> None:
        """Admit a host edit and preserve the refusing scope's measured evidence.

        The nested owner shares the native pool, reservation target, and
        nonrenewing deadline. Its nine work slots retain execution-scope
        semantics rather than borrowing planar predicate-counter identities.
        """
        self._active_native_library("NativeExecutionScope.admit_host_cavity")
        edit = NativeExecutionScope(self._limits_record)
        try:
            with edit:
                edit.admit_cavity(count)
        except NativeResourceFailure:
            raise
        except MeshcoreError as error:
            if (
                error.status
                not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT)
                or edit.evidence is None
            ):
                raise
            raise NativeResourceFailure(
                edit.completed_evidence(), "execution_admit_host_cavity"
            ) from error

    def host_workspace(self) -> NativeHostStorageWorkspace:
        """Reserve real Python-owned storage against this pool's input cap.

        Payloads remain in their Python owners. Their conservative bound
        coexists with native managed allocations; it is not measured payload.
        """
        return NativeHostStorageWorkspace(self)

    @overload
    def allocate_host_array[ScalarT: np.generic](
        self,
        shape: tuple[int, ...],
        dtype: type[ScalarT] | np.dtype[ScalarT],
    ) -> NDArray[ScalarT]: ...

    @overload
    def allocate_host_array(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> NDArray[np.generic]: ...

    def allocate_host_array(
        self,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> NDArray[np.generic]:
        """Allocate an uninitialized, C-contiguous NumPy view of native bytes.

        Only this allocation's native metadata and payload requests are
        counted, in the same managed pool as native scratch. NumPy's Python
        array/buffer metadata and unrelated NumPy/JAX/compiler allocations are
        not measured. The buffer retains its native owner after scope exit;
        NumPy views keep it alive until the last buffer reference is released.
        """
        library = self._active_native_library("NativeExecutionScope.allocate_host_array")
        if _active_execution_scope.get() is not self:
            raise RuntimeError(
                "NativeExecutionScope.allocate_host_array.scope must be the active scope; got foreign scope."
            )
        if not isinstance(shape, tuple) or any(
            isinstance(size, (bool, np.bool_)) or not isinstance(size, int)
            for size in shape
        ):
            raise TypeError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must be a tuple of nonnegative Python integers; got {_input_summary(shape)}."
            )
        if len(shape) > 64:
            raise ValueError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must satisfy the input contract (shape exceeds NumPy's 64-dimensional array limit.); got {_input_summary(shape)}."
            )
        address_limit = np.iinfo(np.intp).max
        if any(size < 0 or size > address_limit for size in shape):
            raise ValueError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must be nonnegative and addressable; got {_input_summary(shape)}."
            )
        if dtype is None:
            raise TypeError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.dtype must be explicit; got {_input_summary(dtype)}."
            )
        element_dtype = np.dtype(dtype)
        if element_dtype.hasobject:
            raise TypeError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.dtype must satisfy the input contract (Native host arrays cannot contain object dtype.); got {_input_summary(dtype)}."
            )
        if element_dtype.subdtype is not None:
            raise ValueError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must satisfy the input contract (A top-level subarray dtype would change the requested shape.); got {_input_summary(shape)}."
            )
        # Empty dimensions still need representable C strides; the
        # payload request below remains exactly zero for an empty array.
        if math.prod(max(1, size) for size in shape) > (
            address_limit // max(1, element_dtype.itemsize)
        ):
            raise ValueError(
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must satisfy the input contract (shape and dtype exceed addressable NumPy array storage.); got {_input_summary(shape)}."
            )
        byte_count = math.prod(shape) * element_dtype.itemsize
        owner = ctypes.c_void_p()
        data = ctypes.c_void_p()
        observation_ticket = self._managed_observation_target()
        with self._managed_retained_native_call():
            _raise_for_call(
                library["phx_mc_execution_allocate_host_array"](
                    self._handle,
                    byte_count,
                    max(1, element_dtype.alignment),
                    ctypes.addressof(owner),
                    ctypes.addressof(data),
                ),
                "execution_allocate_host_array",
            )
        self._managed_observation_mutation(observation_ticket)
        self._managed_observation_candidate()
        if owner.value is None or data.value is None:
            library["phx_mc_execution_free_host_array"](owner)
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.NativeExecutionScope.allocate_host_array.shape must satisfy the input contract (execution_allocate_host_array: missing buffer); got {_input_summary(shape)}.",
            )
        try:
            buffer = (ctypes.c_ubyte * byte_count).from_address(data.value)
            finalizer = weakref.finalize(
                buffer,
                _free_host_array,
                library,
                owner.value,
                weakref.ref(self._managed_observer_owner or self),
                observation_ticket,
                weakref.ref(self),
            )
        except BaseException:
            _free_host_array(
                library,
                owner.value,
                weakref.ref(self._managed_observer_owner or self),
                observation_ticket,
                weakref.ref(self),
            )
            raise
        try:
            # Interpreter atexit callbacks may still hold and use array views;
            # only buffer collection may release its native storage.
            finalizer.atexit = False
            # ctypes arrays expose an instance dictionary absent from their stubs.
            buffer._native_execution_finalizer = finalizer  # ty: ignore[unresolved-attribute]
            return np.ndarray(shape, dtype=element_dtype, buffer=buffer)
        except BaseException:
            finalizer()
            raise

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._closed_by_owner and self._thread is None:
            return
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError(
                f"_meshcore.NativeExecutionScope.__exit__.error_type must end on its creating thread; got {_input_summary(error_type)}."
            )
        primary = _close_native_context_children(self, error)
        cleanup_error = primary if error is None else None
        if self._preparation_attachment is not None:
            try:
                self._preparation_attachment.close()
            except BaseException as failure:
                if primary is None:
                    cleanup_error = primary = failure
                else:
                    primary.add_note(
                        f"Native preparation attachment cleanup also failed: {failure}"
                    )
            finally:
                self._preparation_attachment = None
        for view in reversed(tuple(self._ticket_views)):
            try:
                view.__exit__(error_type, primary, traceback)
            except BaseException as failure:
                if primary is None:
                    cleanup_error = primary = failure
                else:
                    primary.add_note(f"Native scope-view cleanup also failed: {failure}")
        for ticket in reversed(tuple(self._tickets)):
            try:
                ticket.__exit__(error_type, primary, traceback)
            except BaseException as failure:
                if primary is None:
                    cleanup_error = primary = failure
                else:
                    primary.add_note(f"Native reservation cleanup also failed: {failure}")
        primary = error if error is not None else cleanup_error
        if primary is not None:
            self._close_source_metadata()
        if isinstance(primary, MeshcoreError):
            # Direct scope refusals carry the same immutable first-refusal
            # snapshot as ticket refusals; the handle must still be live.
            try:
                _attach_scope_refusal(self._library, self._handle, primary)
            except BaseException as failure:
                primary.add_note(f"Native refusal evidence query also failed: {failure}")
        self._managed_observation_candidate()
        counters = np.zeros((9,), dtype=np.uint64)
        memory = np.zeros((8,), dtype=np.uint64)
        seconds = np.zeros((1,), dtype=np.float64)
        prior_seconds = np.zeros((1,), dtype=np.float64)
        _raise_for_call(
            self._library["phx_mc_execution_prior_seconds"](
                self._handle,
                prior_seconds.ctypes.data,
            ),
            "execution_prior_seconds",
        )
        status = MeshcoreStatus(
            self._library["phx_mc_execution_end"](
                self._handle,
                counters.ctypes.data,
                memory.ctypes.data,
                seconds.ctypes.data,
            )
        )
        if status == MeshcoreStatus.INVALID_ARGUMENT:
            try:
                _raise_for_call(status, "execution_end")
            except BaseException as validation_error:
                if primary is None:
                    raise
                primary.add_note(
                    "Native execution end validation also refused: INVALID_ARGUMENT"
                )
                raise primary from validation_error
        self._handle = ctypes.c_void_p()
        self._thread = None
        _native_context_stack.set(_native_context_stack.get()[:-1])
        if self._token is not None:
            _active_execution_scope.reset(self._token)
            self._token = None
        if self._managed_observation_window_active:
            owner = self._managed_observer_owner
            assert owner is not None
            if owner._managed_observation_window is not self:
                owner._managed_observation_complete = False
            owner._managed_observation_window = self._managed_observation_previous_window
            self._managed_observation_window_active = False
        if status == MeshcoreStatus.OK and isinstance(primary, MeshcoreError):
            status = primary.status
        work = NativeExecutionWorkCounters.from_slots(counters[:6])
        allocation = NativeExecutionMemoryCounters.from_slots(memory[:6])
        self.evidence = NativeExecutionEvidence(
            work,
            allocation,
            float(seconds[0]),
            status,
            int(counters[6]),
            int(counters[7]),
            int(counters[8]),
            int(memory[6]),
            int(memory[7]),
            self._managed_observation_result(int(memory[2]), int(memory[7])),
            float(prior_seconds[0]),
        )
        observation_parent = self._managed_observation_parent
        if (
            observation_parent is not None
            and observation_parent._managed_observation_enabled
        ):
            measured = self.evidence.managed_admission_peak_bytes_upper
            if measured is None:
                observation_parent._managed_observation_complete = False
            else:
                observation_parent._managed_observation_peak = max(
                    observation_parent._managed_observation_peak, measured
                )
        self._source_metadata_completed = primary is None and status is MeshcoreStatus.OK
        if not self._source_metadata_completed:
            self._close_source_metadata()
        if isinstance(primary, MeshcoreError):
            if (
                isinstance(primary, NativeResourceFailure)
                and primary.execution_evidence is not None
            ):
                primary.work_counters = primary.execution_evidence.work_counters
                primary.memory_counters = primary.execution_evidence.memory_counters
            else:
                primary.work_counters = work
                primary.memory_counters = allocation
            if primary.status is MeshcoreStatus.TIMEOUT:
                primary.elapsed_seconds = float(seconds[0])
                primary.maximum_seconds = self._effective_seconds
            if status is primary.status and self._parent is not None:
                # A nested scope shares its parent's native pool, whose terminal
                # status remains visible when the parent later ends. The child
                # exception is the consumer boundary for this refusal; retain
                # the parent's non-OK evidence without raising it a second time.
                self._parent._surfaced_child_statuses.add(status)
        if cleanup_error is not None:
            raise cleanup_error
        if (
            error is None
            and status != MeshcoreStatus.OK
            and status not in self._surfaced_child_statuses
        ):
            raise MeshcoreError(
                status,
                f"_meshcore.NativeExecutionScope.__exit__.error_type must satisfy the input contract (native execution allowance exhausted); got {_input_summary(error_type)}.",
                work_counters=work,
                memory_counters=allocation,
                elapsed_seconds=(
                    float(seconds[0]) if status is MeshcoreStatus.TIMEOUT else None
                ),
                maximum_seconds=(
                    self._effective_seconds if status is MeshcoreStatus.TIMEOUT else None
                ),
            )

    def _close_source_metadata(self) -> None:
        """Release incomplete publication tokens without owning their wrappers."""
        leases, self._source_metadata = self._source_metadata, []
        for reference in leases:
            storage = reference()
            if storage is not None:
                storage.close()


@final
class NativeScopeTicket:
    """Lazy host-accounting view of one actual native request.

    No work, query, cavity or byte allowance is reserved at construction. Host
    work is charged directly to the active descendant of ``scope``; native
    owners keep using that same request, without an ambient reservation target.
    Each managed contributor owns only its actual live bound. The view's local
    counters exclude native owners and imported preparation, already accounted
    by the actual scope.
    """

    def __init__(self, scope: NativeExecutionScope, /) -> None:
        self.scope = scope
        self.work_upper = scope.limits.maximum_work_units
        self.queries_upper = scope.limits.maximum_geometry_queries
        self.cavity_upper = scope.limits.maximum_cavity_cells
        self.work_units = 0
        self.geometry_queries = 0
        self.cavity_cells = 0
        self.managed_bytes = 0
        self.settled = False
        self.native_accounting = True
        self._entered = False
        self._default_bytes = 0
        self._components: dict[ManagedBytesContributor, int] = {}
        self._workspaces: dict[NativeBatchTicket | None, NativeHostStorageWorkspace] = {}
        self._active_scope()
        scope._ticket_views[self] = None

    def _active_scope(self) -> NativeExecutionScope:
        if self.settled:
            raise RuntimeError("NativeScopeTicket must remain live; got settled view.")
        active = current_native_execution_scope()
        owner = active
        while owner is not None and owner is not self.scope:
            owner = owner.parent
        if owner is None or active is None:
            raise RuntimeError(
                "NativeScopeTicket must use its actual scope or a live descendant; got foreign scope."
            )
        return active

    def _remaining(self) -> _RemainingValues:
        active = self._active_scope()
        remaining = active._remaining_values()
        target = active._target_for_thread()
        if target is None:
            # Native remaining already intersects a target borrowed when this
            # nested scope entered with its own local limits.
            return remaining
        # A target installed after scope entry owns genuine reserved capacity.
        # The scope's unreserved observation intentionally excludes that escrow,
        # whereas charge/admit/allocation dispatch to this live target.
        return _RemainingValues(
            target.remaining_work_units,
            target.remaining_geometry_queries,
            target.remaining_cavity_cells,
            target.remaining_managed_bytes,
            remaining.remaining_seconds,
            remaining.status,
        )

    def _storage_workspace(self) -> NativeHostStorageWorkspace:
        active = self._active_scope()
        target = active._target_for_thread()
        if target is None:
            target = active._borrowed_ticket
        workspace = self._workspaces.get(target)
        if workspace is None:
            workspace = active.host_workspace().open()
            self._workspaces[target] = workspace
        return workspace

    @property
    def _live_bytes(self) -> int:
        return self._default_bytes + sum(self._components.values())

    @property
    def bytes_upper(self) -> int:
        return self._live_bytes + self._remaining().remaining_managed_bytes

    @property
    def remaining_work_units(self) -> int:
        return self._remaining().remaining_work_units

    @property
    def remaining_geometry_queries(self) -> int:
        return self._remaining().remaining_geometry_queries

    @property
    def remaining_cavity_cells(self) -> int:
        return self._remaining().remaining_cavity_cells

    @property
    def remaining_managed_bytes(self) -> int:
        """Current unconsumed capacity, excluding this view's same-owner credit."""
        return self._remaining().remaining_managed_bytes

    def _set_bytes(self, bound: int, /) -> None:
        self._active_scope()
        live = sum(workspace.bound for workspace in self._workspaces.values())
        if bound > live:
            # Retained bytes from earlier phases remain charged where earned.
            # Only new live capacity borrows the current target; native storage
            # tokens cannot grow under a foreign target.
            workspace = self._storage_workspace()
            workspace.set_bound(workspace.bound + bound - live)
        elif bound < live:
            release = live - bound
            for workspace in reversed(tuple(self._workspaces.values())):
                amount = min(release, workspace.bound)
                if amount:
                    workspace.set_bound(workspace.bound - amount)
                    release -= amount
                if not release:
                    break
        self.managed_bytes = max(self.managed_bytes, bound)

    def managed_storage(self) -> ManagedBytesContributor:
        self._active_scope()
        return ManagedBytesContributor(self)

    def set_managed_storage(
        self, owner: ManagedBytesContributor, bound: int | None, /
    ) -> None:
        if owner.ticket is not self:
            raise RuntimeError(
                "NativeScopeTicket contributor must belong to this view; got foreign owner."
            )
        if self.settled and bound is None:
            return
        proposed = 0 if bound is None else _unsigned_budget(bound, "managed_bytes")
        self._set_bytes(self._live_bytes - self._components.get(owner, 0) + proposed)
        if bound is None:
            self._components.pop(owner, None)
        else:
            self._components[owner] = proposed

    def managed_storage_headroom(self, owner: ManagedBytesContributor, /) -> int:
        if owner.ticket is not self:
            raise RuntimeError(
                "NativeScopeTicket contributor must belong to this view; got foreign owner."
            )
        # Admit the real current-target token before reporting exact payload
        # headroom. Its actual metadata coexists with all earlier live bounds.
        self._storage_workspace().reserve_empty()
        return self._components.get(owner, 0) + self._remaining().remaining_managed_bytes

    def consume(
        self,
        work: int = 0,
        /,
        *,
        geometry_queries: int = 0,
        cavity_cells: int | None = None,
        managed_bytes: int | None = None,
    ) -> None:
        work = _unsigned_budget(work, "work")
        queries = _unsigned_budget(geometry_queries, "geometry_queries")
        cavity = (
            None
            if cavity_cells is None
            else _unsigned_budget(cavity_cells, "cavity_cells")
        )
        managed = (
            None
            if managed_bytes is None
            else _unsigned_budget(managed_bytes, "managed_bytes")
        )
        active = self._active_scope()
        target = active._target_for_thread()
        if target is None:
            target = active._borrowed_ticket
        if target is not None and (
            work > target.remaining_work_units
            or queries > target.remaining_geometry_queries
        ):
            target.refuse(work, queries_upper=queries)
        if cavity is not None or managed is not None:
            remaining = self._remaining()
            if (
                work > remaining.remaining_work_units
                or queries > remaining.remaining_geometry_queries
            ):
                self.refuse(
                    work,
                    queries_upper=queries,
                    cavity_upper=0 if cavity is None else cavity,
                )
        if cavity is not None:
            active.admit_cavity(cavity)
            self.cavity_cells = max(self.cavity_cells, cavity)
        if managed is not None:
            self._set_bytes(self._live_bytes - self._default_bytes + managed)
            self._default_bytes = managed
        active.charge(work=work, geometry_queries=queries)
        self.work_units += work
        self.geometry_queries += queries

    def batch(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NativeBatchTicket:
        return self._active_scope().batch(
            work_upper,
            queries_upper=queries_upper,
            bytes_upper=bytes_upper,
            cavity_upper=cavity_upper,
        )

    def checkpoint(self) -> None:
        self._active_scope().charge()

    def refuse(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NoReturn:
        work = _unsigned_budget(work_upper, "work_upper")
        queries = _unsigned_budget(queries_upper, "queries_upper")
        cavity = _unsigned_budget(cavity_upper, "cavity_upper")
        managed = _unsigned_budget(bytes_upper, "bytes_upper")
        active = self._active_scope()
        remaining = self._remaining()
        # Only invoke a consuming admission when it is already proven to
        # refuse. Fitting estimates must never debit the actual request.
        if (
            work > remaining.remaining_work_units
            or queries > remaining.remaining_geometry_queries
        ):
            target = active._target_for_thread()
            if target is None:
                target = active._borrowed_ticket
            if target is not None:
                target.refuse(work, queries_upper=queries)
            active.charge(work=work, geometry_queries=queries)
        elif cavity > remaining.remaining_cavity_cells:
            active.admit_cavity(cavity)
        elif managed > self._live_bytes + remaining.remaining_managed_bytes:
            self._set_bytes(managed)
        else:
            self.checkpoint()
        raise RuntimeError(
            "NativeScopeTicket.refuse must prove an actual resource shortage; got a fitting estimate."
        )

    @contextmanager
    def native_target(self) -> Iterator[None]:
        # Check at entry, not when the context object is created: moving this
        # same-thread view to a worker must not permit unscoped native work.
        active = self._active_scope()
        owner = active._managed_observer_owner or active
        if (
            not owner._managed_observation_enabled
            or active._target_for_thread() is not None
        ):
            yield
            return
        if active._borrowed_ticket is not None:
            if active._borrowed_target_for_thread() is None:
                owner._managed_native_events_complete = False
                raise RuntimeError(
                    "NativeScopeTicket.native_target must borrow a live ancestor target "
                    "on its owning thread; got an inactive or foreign target."
                )
            if not active._managed_observation_enabled:
                yield
                return
        remaining = self._remaining()
        with active.batch(
            remaining.remaining_work_units,
            queries_upper=remaining.remaining_geometry_queries,
            cavity_upper=remaining.remaining_cavity_cells,
            bytes_upper=remaining.remaining_managed_bytes,
        ) as phase:
            if not phase._managed_observation_phase:
                # A real zero-work FFI phase is not a retained cache ticket.
                phase._managed_observation_phase = True
                active._managed_observation_admit(phase)
            with phase.native_target():
                yield

    def settle(self) -> None:
        self._active_scope()
        for workspace in reversed(tuple(self._workspaces.values())):
            workspace.close()
        self._workspaces.clear()
        self._components.clear()
        self._default_bytes = 0
        self.settled = True
        del self.scope._ticket_views[self]

    def __enter__(self) -> NativeScopeTicket:
        self._active_scope()
        if self._entered:
            raise RuntimeError("NativeScopeTicket must be single-use; got entered view.")
        self._entered = True
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.settled:
            return
        try:
            self.settle()
        except BaseException as release:
            if error is None:
                raise
            error.add_note(f"Native scope-view release also failed: {release}")


@final
class NativeBatchTicket:
    """A reservation shared by local host counters and explicit native targets.

    Own work includes host, native and imported charges. Consumed work also
    includes settled descendants; native-only counters exclude host/imports.
    Capacity properties retain simultaneous peaks after settlement.
    Context exit retires children and targets before settlement; explicit
    ``settle`` requires their cleanup first. A refusal with live children
    records native refusal evidence immediately, admits only releases, and
    settles when the last child settles.
    """

    def __init__(
        self,
        scope: NativeExecutionScope,
        parent: NativeBatchTicket | None,
        work: int,
        queries: int,
        cavity: int,
        managed: int,
    ) -> None:
        if scope._library is None or scope._thread is None:
            raise RuntimeError(
                "NativeBatchTicket.scope must be live; got inactive scope."
            )
        if any(value is None for value in (work, queries, cavity, managed)):
            raise TypeError(
                "NativeBatchTicket.upper_bounds must be explicit integers; got None."
            )
        self.scope = scope
        self.parent = parent
        self.work_upper = _unsigned_budget(work, "work_upper")
        self.queries_upper = _unsigned_budget(queries, "queries_upper")
        self.cavity_upper = _unsigned_budget(cavity, "cavity_upper")
        self.bytes_upper = _unsigned_budget(managed, "bytes_upper")
        self._managed_observation_phase = bool(
            self.work_upper or self.queries_upper or self.cavity_upper
        )
        self._usage = _NativeExecutionUsage()
        self._mapped = False
        self._usage_thread: int | None = None
        self._peak_cavity = 0
        self._peak_bytes = 0
        self._managed_admission_peak = 0
        self._default_managed_bytes = 0
        self._managed_components: dict[ManagedBytesContributor, int] = {}
        self._children: set[NativeBatchTicket] = set()
        self._lock = threading.RLock()
        self._handle = ctypes.c_void_p()
        self._library = scope._library
        self._entered = False
        self._settled = False
        self._refused = False
        self._thread: int | None = None
        self._target: _NativeBatchTarget | None = None
        if parent is None:
            self._admit()
        else:
            with parent._lock:
                parent._require_unrefused()
                parent._map_usage()
                if (
                    self.work_upper > parent.remaining_work_units
                    or self.queries_upper > parent.remaining_geometry_queries
                    or self.cavity_upper > parent.remaining_cavity_cells
                    or self.bytes_upper > parent.remaining_managed_bytes
                ):
                    # The canonical refusal operation distinguishes an actual
                    # root/ancestor shortage from a fitting estimate defect.
                    parent.refuse(
                        self.work_upper,
                        queries_upper=self.queries_upper,
                        cavity_upper=self.cavity_upper,
                        bytes_upper=self.bytes_upper,
                    )
                self._admit()
                parent._children.add(self)
                parent._observe_managed_admission_capacity()
        scope._tickets[self] = None
        scope._managed_observation_admit(self)
        scope._managed_observation_candidate()

    def _admit(self) -> None:
        status = MeshcoreStatus(
            self._library["phx_mc_execution_batch_admit"](
                self.scope._handle,
                None if self.parent is None else self.parent._handle,
                self.work_upper,
                self.queries_upper,
                self.cavity_upper,
                self.bytes_upper,
                ctypes.addressof(self._handle),
            )
        )
        if status is not MeshcoreStatus.OK:
            raise self._failure(
                status,
                "execution_batch_admit",
                (
                    self.work_upper,
                    self.queries_upper,
                    self.cavity_upper,
                    self.bytes_upper,
                ),
            )

    def _require_live(self) -> None:
        if self._settled:
            raise RuntimeError(
                f"_meshcore.NativeBatchTicket._require_live.self must remain unsettled; got settled ticket; got {_input_summary(self)}."
            )

    def _require_unrefused(self) -> None:
        self._require_live()
        if self._refused:
            raise RuntimeError(
                f"_meshcore.NativeBatchTicket._require_unrefused.self must remain unrefused; got a refused ticket awaiting child settlement; got {_input_summary(self)}."
            )

    def _require_host_writer(self) -> None:
        if self._mapped and self._usage_thread != threading.get_ident():
            raise RuntimeError(
                "NativeBatchTicket.host_counters must use their affine owner; got another thread."
            )

    def _install_usage(self, address: int, /) -> None:
        if address % ctypes.alignment(_NativeExecutionUsage):
            raise RuntimeError(
                "NativeBatchTicket.usage must be uint64-aligned; got an unaligned view."
            )
        if self._mapped:
            if ctypes.addressof(self._usage) != address:
                raise RuntimeError(
                    "NativeBatchTicket.usage must retain its mapped identity; got another view."
                )
            return
        previous = self._usage
        usage = _NativeExecutionUsage.from_address(address)
        usage.host_work = previous.host_work
        usage.host_queries = previous.host_queries
        usage.host_cavity = previous.host_cavity
        usage.host_bytes = previous.host_bytes
        usage.host_peak_cavity = previous.host_peak_cavity
        usage.host_peak_bytes = previous.host_peak_bytes
        self._usage = usage
        self._mapped = True
        self._usage_thread = threading.get_ident()

    def _map_usage(self) -> None:
        if self._mapped:
            return
        owner = self._thread if self._thread is not None else self.scope._thread
        if owner != threading.get_ident():
            raise RuntimeError(
                "NativeBatchTicket.usage must first map on its host owner; got another thread."
            )
        address = ctypes.c_void_p()
        _raise_for_call(
            self._library["phx_mc_execution_batch_usage"](
                self._handle, ctypes.addressof(address)
            ),
            "execution_batch_usage",
        )
        if address.value is None:
            raise RuntimeError(
                "NativeBatchTicket.usage must map a live view; got a null view."
            )
        self._install_usage(address.value)

    def _snapshot(self) -> _NativeExecutionUsage:
        usage = self._usage
        # Scalar reads preserve the shared view's naturally aligned uint64 loads.
        return _NativeExecutionUsage(
            host_work=usage.host_work,
            host_queries=usage.host_queries,
            host_cavity=usage.host_cavity,
            host_bytes=usage.host_bytes,
            host_peak_cavity=usage.host_peak_cavity,
            host_peak_bytes=usage.host_peak_bytes,
            native_work=usage.native_work,
            native_queries=usage.native_queries,
            native_cavity=usage.native_cavity,
            native_bytes=usage.native_bytes,
            native_peak_cavity=usage.native_peak_cavity,
            native_peak_bytes=usage.native_peak_bytes,
            reserved_work=usage.reserved_work,
            reserved_queries=usage.reserved_queries,
            reserved_cavity=usage.reserved_cavity,
            reserved_bytes=usage.reserved_bytes,
            children_work=usage.children_work,
            children_queries=usage.children_queries,
            children_peak_cavity=usage.children_peak_cavity,
            children_peak_bytes=usage.children_peak_bytes,
            imported_work=usage.imported_work,
            imported_queries=usage.imported_queries,
            children_native_work=usage.children_native_work,
            children_native_queries=usage.children_native_queries,
            children_live_bytes=usage.children_live_bytes,
            retained_host_bytes=usage.retained_host_bytes,
            combined_peak_cavity=usage.combined_peak_cavity,
            combined_peak_bytes=usage.combined_peak_bytes,
        )

    def _attach_refusal(self, error: MeshcoreError, /) -> None:
        if error.status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
            return
        if self._handle.value is not None and error.batch_admitted_upper is None:
            error.batch_admitted_upper = (
                self.work_upper,
                self.queries_upper,
                self.cavity_upper,
                self.bytes_upper,
            )
        _attach_scope_refusal(self._library, self.scope._handle, error)

    def _failure(
        self,
        status: MeshcoreStatus,
        operation: str,
        requested: tuple[int, int, int, int] | None = None,
    ) -> MeshcoreError:
        error = MeshcoreError(
            status, f"{operation} must fit the native allowance; got {status.name}."
        )
        error.batch_requested_upper = requested
        self._attach_refusal(error)
        return error

    @property
    def settled(self) -> bool:
        """Whether measured consumption was committed and the reservation released."""
        return self._settled

    @property
    def native_accounting(self) -> bool:
        return True

    @property
    def work_units(self) -> int:
        usage = self._usage
        return usage.host_work + usage.native_work + usage.imported_work

    @property
    def geometry_queries(self) -> int:
        usage = self._usage
        return usage.host_queries + usage.native_queries + usage.imported_queries

    @property
    def consumed_work_units(self) -> int:
        return self.work_units + self._usage.children_work

    @property
    def consumed_geometry_queries(self) -> int:
        return self.geometry_queries + self._usage.children_queries

    @property
    def native_work_units(self) -> int:
        return self._usage.native_work + self._usage.children_native_work

    @property
    def native_geometry_queries(self) -> int:
        return self._usage.native_queries + self._usage.children_native_queries

    @property
    def remaining_work_units(self) -> int:
        usage = self._usage
        return max(
            0,
            self.work_upper
            - usage.host_work
            - usage.native_work
            - usage.imported_work
            - usage.children_work
            - usage.reserved_work,
        )

    @property
    def remaining_geometry_queries(self) -> int:
        usage = self._usage
        return max(
            0,
            self.queries_upper
            - usage.host_queries
            - usage.native_queries
            - usage.imported_queries
            - usage.children_queries
            - usage.reserved_queries,
        )

    @property
    def remaining_cavity_cells(self) -> int:
        usage = self._usage
        return max(
            0,
            self.cavity_upper
            - usage.host_cavity
            - usage.native_cavity
            - usage.reserved_cavity,
        )

    @property
    def remaining_managed_bytes(self) -> int:
        usage = self._usage
        return max(
            0,
            self.bytes_upper
            - usage.host_bytes
            - usage.native_bytes
            - usage.retained_host_bytes
            - usage.children_live_bytes
            - usage.reserved_bytes,
        )

    def _live_capacity(self) -> tuple[int, int]:
        usage = self._usage
        cavity = usage.host_cavity + usage.native_cavity
        managed = (
            usage.host_bytes
            + usage.native_bytes
            + usage.retained_host_bytes
            + usage.children_live_bytes
        )
        if self._children:
            for child in tuple(self._children):
                child_cavity, child_bytes = child._live_capacity()
                cavity += child_cavity
                managed += child_bytes
        return cavity, managed

    def _observe_capacity(self) -> None:
        usage = self._usage
        if self._children:
            cavity, managed = self._live_capacity()
        else:
            # A leaf needs no recursive result tuple. Read the mapped native
            # fields on every observation; only host fields have an affine writer.
            cavity = usage.host_cavity + usage.native_cavity
            managed = (
                usage.host_bytes
                + usage.native_bytes
                + usage.retained_host_bytes
                + usage.children_live_bytes
            )
        self._peak_cavity = max(
            self._peak_cavity,
            cavity,
            usage.host_peak_cavity,
            usage.native_peak_cavity,
            usage.children_peak_cavity,
            usage.combined_peak_cavity,
        )
        self._peak_bytes = max(
            self._peak_bytes,
            managed,
            usage.host_peak_bytes,
            usage.native_peak_bytes,
            usage.children_peak_bytes,
            usage.combined_peak_bytes,
        )
        self._observe_managed_admission_capacity()
        self.scope._managed_observation_candidate()

    def _observe_managed_admission_capacity(self) -> None:
        owner = self.scope._managed_observer_owner or self.scope
        if not owner._managed_observation_enabled:
            return
        usage = self._usage
        # Each live child's entire admitted upper is already in reserved_bytes.
        # Adding that child's current actual usage would count it twice.
        simultaneous = (
            usage.host_bytes
            + usage.native_bytes
            + usage.retained_host_bytes
            + usage.children_live_bytes
            + usage.reserved_bytes
        )
        self._managed_admission_peak = max(
            self._managed_admission_peak,
            self._peak_bytes,
            usage.combined_peak_bytes,
            simultaneous,
        )

    @property
    def cavity_cells(self) -> int:
        with self._lock:
            self._observe_capacity()
            return self._peak_cavity

    @property
    def managed_bytes(self) -> int:
        with self._lock:
            self._observe_capacity()
            return self._peak_bytes

    def _observe_descendant_capacity(self) -> None:
        with self._lock:
            self._observe_capacity()
        if self.parent is not None:
            self.parent._observe_descendant_capacity()

    def __enter__(self) -> NativeBatchTicket:
        with self._lock:
            self._require_unrefused()
            self._require_host_writer()
            if self._entered:
                raise RuntimeError(
                    f"_meshcore.NativeBatchTicket.__enter__.self must be single-use; got entered ticket; got {_input_summary(self)}."
                )
            self._entered = True
            self._thread = threading.get_ident()
        return self

    def consume(
        self,
        work: int = 0,
        /,
        *,
        geometry_queries: int = 0,
        cavity_cells: int | None = None,
        managed_bytes: int | None = None,
    ) -> None:
        """Charge own host work and replace optional live host capacities.

        The default managed component and explicit contributors are additive.
        Host estimate defects do not mutate counters or refuse native resources.
        """
        work = _unsigned_budget(work, "work")
        queries = _unsigned_budget(geometry_queries, "geometry_queries")
        with self._lock:
            self._require_unrefused()
            self._require_host_writer()
            usage = self._usage
            cavity = (
                usage.host_cavity
                if cavity_cells is None
                else _unsigned_budget(cavity_cells, "cavity_cells")
            )
            direct = (
                self._default_managed_bytes
                if managed_bytes is None
                else _unsigned_budget(managed_bytes, "managed_bytes")
            )
            # host_bytes is the affine writer's current aggregate. Native
            # capacity has separate fields and must still be read below.
            managed = (
                usage.host_bytes
                if managed_bytes is None
                else usage.host_bytes - self._default_managed_bytes + direct
            )
            if (
                work > self.remaining_work_units
                or queries > self.remaining_geometry_queries
                or cavity
                > max(
                    usage.host_cavity,
                    self.cavity_upper - usage.native_cavity - usage.reserved_cavity,
                )
                or managed
                > max(
                    usage.host_bytes,
                    self.bytes_upper
                    - usage.native_bytes
                    - usage.retained_host_bytes
                    - usage.children_live_bytes
                    - usage.reserved_bytes,
                )
            ):
                raise RuntimeError(
                    "NativeBatchTicket.consume must fit the admitted upper bound; got estimate defect."
                )
            usage.host_work += work
            usage.host_queries += queries
            usage.host_cavity = cavity
            usage.host_bytes = managed
            usage.host_peak_cavity = max(usage.host_peak_cavity, cavity)
            usage.host_peak_bytes = max(usage.host_peak_bytes, managed)
            self._default_managed_bytes = direct
            self._observe_capacity()
        if self.parent is not None:
            self.parent._observe_descendant_capacity()

    def native_target(self) -> AbstractContextManager[None]:
        """Install this reservation as the native execution and allocation owner."""
        self.scope._managed_observation_activate(self)
        return _NativeBatchTarget(self)

    def managed_storage(self) -> ManagedBytesContributor:
        self._require_unrefused()
        return ManagedBytesContributor(self)

    def set_managed_storage(
        self, owner: ManagedBytesContributor, bound: int | None, /
    ) -> None:
        """Replace or release one identity-checked host-capacity contribution."""
        if owner.ticket is not self:
            raise ValueError(
                "NativeBatchTicket.managed_storage must belong to this ticket; got another owner."
            )
        value = 0 if bound is None else _unsigned_budget(bound, "managed_bytes")
        with self._lock:
            previous = self._managed_components.get(owner, 0)
            if (self._settled or self._refused) and value > previous:
                raise RuntimeError(
                    "NativeBatchTicket.managed_storage must not grow after refusal or settlement; got a larger bound."
                )
            if not self._settled:
                self._require_host_writer()
            if (
                bound is not None
                and owner in self._managed_components
                and value == previous
            ):
                return
            if not self._settled:
                usage = self._usage
                # Contributor replacement changes only its delta in the
                # affine host aggregate, not any native/retained capacity.
                aggregate = usage.host_bytes - previous + value
                if aggregate > max(
                    usage.host_bytes,
                    self.bytes_upper
                    - usage.native_bytes
                    - usage.retained_host_bytes
                    - usage.children_live_bytes
                    - usage.reserved_bytes,
                ):
                    self.refuse(0, bytes_upper=aggregate)
                usage.host_bytes = aggregate
                usage.host_peak_bytes = max(usage.host_peak_bytes, aggregate)
            if bound is None:
                self._managed_components.pop(owner, None)
            else:
                self._managed_components[owner] = value
            if not self._settled:
                self._observe_capacity()
        if not self._settled and self.parent is not None:
            self.parent._observe_descendant_capacity()

    def managed_storage_headroom(self, owner: ManagedBytesContributor, /) -> int:
        """Return the largest bound ``owner`` can take now without a refusal."""
        if owner.ticket is not self:
            raise ValueError(
                "NativeBatchTicket.managed_storage must belong to this ticket; got another owner."
            )
        with self._lock:
            previous = self._managed_components.get(owner, 0)
            if self._settled or self._refused:
                return previous
            usage = self._usage
            # set_managed_storage refuses host - previous + bound above
            # max(host, capacity); this is the same comparison solved for bound.
            capacity = (
                self.bytes_upper
                - usage.native_bytes
                - usage.retained_host_bytes
                - usage.children_live_bytes
                - usage.reserved_bytes
            )
            return previous + max(0, capacity - usage.host_bytes)

    def batch(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NativeBatchTicket:
        """Reserve an explicit child, which may install a target on a worker."""
        return NativeBatchTicket(
            self.scope, self, work_upper, queries_upper, cavity_upper, bytes_upper
        )

    def checkpoint(self) -> None:
        """Check the native deadline without a per-item accounting FFI call."""
        with self._lock:
            self._require_unrefused()
            status = MeshcoreStatus(
                self._library["phx_mc_execution_batch_checkpoint"](self._handle)
            )
            self._observe_capacity()
            if status is not MeshcoreStatus.OK:
                raise self._failure(status, "execution_batch_checkpoint")

    def _require_settleable(self) -> None:
        self._require_live()
        if self._children or self._target is not None:
            raise RuntimeError(
                "NativeBatchTicket.settle must follow child and target cleanup; got live children or target."
            )

    def _host_actual(self) -> tuple[int, int, int, int]:
        usage = self._usage
        return (
            usage.host_work,
            usage.host_queries,
            usage.host_peak_cavity,
            usage.host_peak_bytes,
        )

    def _finish_settlement(self, snapshot: _NativeExecutionUsage, /) -> None:
        self._usage = snapshot
        self._mapped = False
        self._usage_thread = None
        self._settled = True
        self._handle = ctypes.c_void_p()
        del self.scope._tickets[self]
        self.scope._managed_observation_settle(self)
        if self.parent is not None:
            with self.parent._lock:
                self.parent._children.remove(self)
                self.parent._observe_capacity()
                if self.parent._refused and not self.parent._children:
                    self.parent.settle()
                    return
            if self.parent.parent is not None:
                self.parent.parent._observe_descendant_capacity()

    def settle(self) -> None:
        """Commit own host measurements; native folds its usage and children once."""
        with self._lock:
            self._require_settleable()
            self._observe_capacity()
            snapshot = self._snapshot()
            _raise_for_call(
                self._library["phx_mc_execution_batch_settle"](
                    self._handle, *self._host_actual()
                ),
                "execution_batch_settle",
            )
            self._finish_settlement(snapshot)

    def refuse(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NoReturn:
        """Atomically settle and refuse the next increment or replacement capacity.

        A fitting request is an estimate defect and leaves the ticket live.
        A proven resource refusal destroys the ticket without a probe reservation.
        Live children keep their reservations: native records the refusal now,
        and the last child's settlement commits this ticket once.
        """
        requested = (
            _unsigned_budget(work_upper, "work_upper"),
            _unsigned_budget(queries_upper, "queries_upper"),
            _unsigned_budget(cavity_upper, "cavity_upper"),
            _unsigned_budget(bytes_upper, "bytes_upper"),
        )
        with self._lock:
            self._require_unrefused()
            if self._target is not None:
                self._target.close(None)
            pending = bool(self._children)
            if not pending:
                self._require_settleable()
            self._observe_capacity()
            snapshot = self._snapshot()
            actual = np.asarray(self._host_actual(), dtype=np.uint64)
            upper = np.asarray(requested, dtype=np.uint64)
            values = np.zeros((8,), dtype=np.uint64)
            status = MeshcoreStatus(
                self._library["phx_mc_execution_batch_refuse"](
                    self._handle,
                    actual.ctypes.data,
                    upper.ctypes.data,
                    values.ctypes.data,
                )
            )
            if status is MeshcoreStatus.INTERNAL_ERROR:
                raise RuntimeError(
                    "NativeBatchTicket.refuse must prove a resource refusal; got a fitting request or estimate defect."
                )
            if status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
                _raise_for_call(status, "execution_batch_refuse")
                raise RuntimeError(
                    "NativeBatchTicket.refuse must return a refusal; got successful status."
                )
            if pending:
                self._refused = True
            else:
                self._finish_settlement(snapshot)
            values.setflags(write=False)
            error = MeshcoreError(
                status,
                f"NativeBatchTicket.refuse: {'request deadline expired' if status is MeshcoreStatus.TIMEOUT else 'request resource allowance exhausted'}; requested next work/query increments and cavity/storage capacities={requested}, remaining={tuple(values[:4])}, consumed work/queries={tuple(values[4:6])}.",
            )
            error.batch_admitted_upper = (
                self.work_upper,
                self.queries_upper,
                self.cavity_upper,
                self.bytes_upper,
            )
            error.batch_requested_upper = requested
            error.batch_refusal_evidence = values
            raise error

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._settled:
            return
        try:
            if self._target is not None:
                self._target.close(error)
            for child in tuple(self._children):
                child.__exit__(error_type, error, traceback)
            with self._lock:
                if not self._settled:
                    self.settle()
        except BaseException as failure:
            if error is None:
                raise
            error.add_note(f"Native batch settlement also failed: {failure}")


@final
class _NativeBatchTarget(AbstractContextManager[None]):
    """Restore native TLS and the Python FFI owner together on every valid exit."""

    def __init__(self, ticket: NativeBatchTicket, /) -> None:
        self.ticket = ticket
        self._handle = ctypes.c_void_p()
        self._token: Token[NativeExecutionScope | None] | None = None
        self._thread: int | None = None
        self._entered = False
        self._closed = False
        self._joined = False

    def __enter__(self) -> None:
        ticket = self.ticket
        identity = threading.get_ident()
        with ticket._lock:
            ticket._require_unrefused()
            ticket._require_host_writer()
            if self._entered:
                raise RuntimeError(
                    "NativeBatchTicket.native_target must be single-use; got an entered context."
                )
            current_scope = _active_execution_scope.get()
            current_target = (
                None if current_scope is None else current_scope._target_for_thread()
            )
            if current_target is None and current_scope is not None:
                current_target = current_scope._borrowed_ticket
            if current_target is ticket and ticket._target is not None:
                self._entered = True
                self._thread = identity
                self._joined = True
                return None
            if ticket._target is not None:
                raise RuntimeError(
                    "NativeBatchTicket.native_target must have one live target; got an entered target."
                )
            if ticket.parent is None and ticket.scope._thread != identity:
                raise RuntimeError(
                    "NativeBatchTicket.native_target must use an explicit child on workers; got a root ticket."
                )
            address = ctypes.c_void_p()
            status = MeshcoreStatus(
                ticket._library["phx_mc_execution_target_begin"](
                    ticket._handle,
                    ctypes.addressof(self._handle),
                    ctypes.addressof(address),
                )
            )
            if status is not MeshcoreStatus.OK:
                raise ticket._failure(status, "execution_target_begin")
            self._entered = True
            self._thread = identity
            try:
                if self._handle.value is None or address.value is None:
                    raise RuntimeError(
                        "NativeBatchTicket.native_target must return a target and usage; got a null view."
                    )
                ticket._install_usage(address.value)
                ticket._target = self
                ticket.scope._target_stacks.setdefault(identity, []).append(ticket)
                self._token = _active_execution_scope.set(ticket.scope)
                _native_context_stack.set((*_native_context_stack.get(), self))
            except BaseException as error:
                status = MeshcoreStatus(
                    ticket._library["phx_mc_execution_target_end"](self._handle)
                )
                self._closed = True
                ticket._target = None
                stack = ticket.scope._target_stacks.get(identity)
                if stack and stack[-1] is ticket:
                    stack.pop()
                    if not stack:
                        del ticket.scope._target_stacks[identity]
                if self._token is not None:
                    _active_execution_scope.reset(self._token)
                    self._token = None
                if status is not MeshcoreStatus.OK:
                    error.add_note(
                        f"Native target initialization cleanup also refused: {status.name}"
                    )
                raise
        return None

    def close(self, error: BaseException | None, /) -> None:
        if self._closed:
            return
        ticket = self.ticket
        with ticket._lock:
            if self._joined:
                current_scope = _active_execution_scope.get()
                current_target = (
                    None if current_scope is None else current_scope._target_for_thread()
                )
                if current_target is None and current_scope is not None:
                    current_target = current_scope._borrowed_ticket
                if self._thread != threading.get_ident() or (
                    not ticket.settled and current_target is not ticket
                ):
                    raise RuntimeError(
                        "NativeBatchTicket.native_target must close a joined context on its current owner; got another target."
                    )
                self._closed = True
                return
            primary = _close_native_context_children(self, error)
            stack = ticket.scope._target_stacks.get(threading.get_ident())
            if (
                self._thread != threading.get_ident()
                or not stack
                or stack[-1] is not ticket
                or _active_execution_scope.get() is not ticket.scope
            ):
                raise RuntimeError(
                    "NativeBatchTicket.native_target must close in owner-thread LIFO order; got another owner."
                )
            ticket._observe_capacity()
            status = MeshcoreStatus(
                ticket._library["phx_mc_execution_target_end"](self._handle)
            )
            if status is MeshcoreStatus.INVALID_ARGUMENT:
                _raise_for_call(status, "execution_target_end")
            self._closed = True
            self._handle = ctypes.c_void_p()
            ticket._target = None
            stack.pop()
            if not stack:
                del ticket.scope._target_stacks[threading.get_ident()]
            if self._token is not None:
                _active_execution_scope.reset(self._token)
                self._token = None
            _native_context_stack.set(_native_context_stack.get()[:-1])
            if isinstance(error, MeshcoreError):
                ticket._attach_refusal(error)
            if status is not MeshcoreStatus.OK:
                if primary is None:
                    raise ticket._failure(status, "execution_target_end")
                primary.add_note(f"Native target cleanup also refused: {status.name}")
            if error is None and primary is not None:
                raise primary

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close(error)
        except BaseException as failure:
            if error is None:
                raise
            error.add_note(f"Native target cleanup also failed: {failure}")


_native_context_stack: ContextVar[
    tuple[NativeExecutionScope | _NativeBatchTarget, ...]
] = ContextVar("native_context_stack", default=())


def _attach_scope_refusal(
    library: MeshcoreLibrary, scope: ctypes.c_void_p, error: MeshcoreError, /
) -> None:
    """Attach a live scope's immutable first-refusal snapshot and request.

    Missing snapshots or requests leave their fields unset.
    """
    if error.status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
        return
    if error.batch_refusal_evidence is None:
        values = np.zeros((8,), dtype=np.uint64)
        observed = MeshcoreStatus(
            library["phx_mc_execution_batch_refusal"](scope, values.ctypes.data)
        )
        if observed is MeshcoreStatus.OK:
            values.setflags(write=False)
            error.batch_refusal_evidence = values
        elif observed is not MeshcoreStatus.INVALID_ARGUMENT:
            _raise_for_call(observed, "execution_batch_refusal")
    if error.batch_requested_upper is None:
        requested = np.zeros((4,), dtype=np.uint64)
        observed = MeshcoreStatus(
            library["phx_mc_execution_refusal_request"](scope, requested.ctypes.data)
        )
        if observed is MeshcoreStatus.OK:
            error.batch_requested_upper = (
                int(requested[0]),
                int(requested[1]),
                int(requested[2]),
                int(requested[3]),
            )
        elif observed is not MeshcoreStatus.INVALID_ARGUMENT:
            _raise_for_call(observed, "execution_refusal_request")


def _close_native_context_children(
    owner: NativeExecutionScope | _NativeBatchTarget,
    error: BaseException | None,
    /,
) -> BaseException | None:
    """Close entered native contexts above their owner in reverse entry order."""
    primary = error
    stack = _native_context_stack.get()
    if owner not in stack:
        raise RuntimeError(
            "NativeExecutionScope.context must close on its active owner thread; "
            "got an unregistered context."
        )
    while stack[-1] is not owner:
        child = stack[-1]
        try:
            match child:
                case NativeExecutionScope():
                    child._closed_by_owner = True
                    child.__exit__(
                        None if primary is None else type(primary), primary, None
                    )
                case _NativeBatchTarget():
                    child.close(primary)
                case invalid:
                    assert_never(invalid)
        except BaseException as failure:
            if primary is None:
                primary = failure
            else:
                primary.add_note(f"Native child-context cleanup also failed: {failure}")
        updated = _native_context_stack.get()
        if updated == stack:
            if primary is not None:
                raise primary
            raise RuntimeError(
                "NativeExecutionScope.context must retire its child before exit; "
                "got an active child."
            )
        stack = updated
    return primary


@final
class NativeHostStorageWorkspace:
    """A real external-host upper bound in the active native allocation pool.

    One opaque token is resized atomically; refusals preserve its old bound.
    Closing releases its bound even after a failed native scope. Explicit
    ``open``/``close`` lifetimes leave the ambient workspace unchanged; context
    entry installs this workspace until exit. No dummy payload is allocated,
    and the native allocator's six measured counters stay separate.
    """

    def __init__(self, budget: NativeExecutionScope, /) -> None:
        self._budget = budget
        self._library: MeshcoreLibrary | None = None
        self._reservation = ctypes.c_void_p()
        self._finalizer: weakref.finalize | None = None
        self._active = False
        self._token: Token[NativeHostStorageWorkspace | None] | None = None
        self.bound = 0
        self._managed_observation_ticket: (
            weakref.ReferenceType[NativeBatchTicket] | None
        ) = None
        self._managed_observation_origin: (
            weakref.ReferenceType[NativeExecutionScope] | None
        ) = None
        self._retained_owners: dict[int, object] = {}
        self._logical_unmanaged_bytes_upper = 0
        self._covered_owners: list[Mapping[int, object]] = []
        self._shared_owners: list[Mapping[int, object]] = []
        self._adopted_reservations: list[
            tuple[NativeBatchTicket, ManagedBytesContributor]
        ] = []
        self._adopted_workspaces: list[NativeHostStorageWorkspace] = []
        self._adopted_by: NativeHostStorageWorkspace | None = None
        self._source_metadata = False

    def open(self) -> NativeHostStorageWorkspace:
        """Open one explicit storage lifetime on the active creating scope."""
        if self._active or self._library is not None:
            raise RuntimeError(
                "NativeHostStorageWorkspace.open must begin a single-use storage lifetime; "
                "got an already opened workspace."
            )
        if (
            current_native_execution_scope() is not self._budget
            or self._budget._library is None
        ):
            raise RuntimeError(
                "NativeHostStorageWorkspace.open must use the active creating scope; "
                "got an inactive or foreign scope."
            )
        self._library = self._budget._library
        self._active = True
        try:
            self.set_bound(0)
        except BaseException:
            self.close()
            raise
        return self

    def __enter__(self) -> NativeHostStorageWorkspace:
        self.open()
        self._token = _active_host_workspace.set(self)
        return self

    def set_bound(self, total_bytes_upper: int, /) -> None:
        self._resize_bound(
            _unsigned_budget(total_bytes_upper, "total_bytes_upper"), reserve_empty=False
        )

    def reserve_empty(self) -> None:
        """Admit this real native token before querying exact payload headroom."""
        self._resize_bound(self.bound, reserve_empty=True)

    def _resize_bound(self, total: int, /, *, reserve_empty: bool) -> None:
        budget = current_native_execution_scope()
        root = budget
        while root is not None and root.parent is not None:
            root = root.parent
        owner_root = self._budget
        while owner_root.parent is not None:
            owner_root = owner_root.parent
        if self._adopted_by is not None:
            raise RuntimeError(
                "NativeHostStorageWorkspace.set_bound must use its current owner; "
                "got a transferred workspace."
            )
        if (
            not self._active
            or self._library is None
            or budget is None
            or budget._library is not self._library
            or root is not owner_root
            or owner_root._thread is None
        ):
            raise RuntimeError(
                "NativeHostStorageWorkspace.set_bound must use its active pool; got inactive or foreign scope."
            )
        if total == 0 and self._reservation.value is None and not reserve_empty:
            return
        if total == self.bound and self._reservation.value is not None:
            return
        observation_ticket = (
            budget._managed_observation_target()
            if self._reservation.value is None
            else self._managed_observation_ticket
        )
        observation_origin = (
            weakref.ref(budget)
            if self._reservation.value is None
            else self._managed_observation_origin
        )
        release = (
            budget._managed_observation_begin_release(
                observation_ticket, origin=observation_origin
            )
            if total < self.bound
            else None
        )
        if total >= self.bound:
            budget._managed_observation_mutation(
                observation_ticket, origin=observation_origin
            )
        try:
            with budget._managed_retained_native_call():
                _raise_for_call(
                    self._library["phx_mc_execution_reserve_host_storage"](
                        budget._handle,
                        total,
                        ctypes.addressof(self._reservation),
                    ),
                    "execution_reserve_host_storage",
                )
        finally:
            if release is not None:
                budget._managed_observation_end_release(release)
        if self._reservation.value is None:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "NativeHostStorageWorkspace.reservation must be present after admission; got null handle.",
            )
        if self._finalizer is None:
            self._finalizer = weakref.finalize(
                self,
                budget._managed_observation_release_host_storage,
                weakref.ref(budget._managed_observer_owner or budget),
                self._library,
                self._reservation.value,
                observation_ticket,
                observation_origin,
            )
            self._finalizer.atexit = False
        self.bound = total
        self._managed_observation_ticket = observation_ticket
        self._managed_observation_origin = observation_origin
        budget._managed_observation_candidate()

    @property
    def logical_unmanaged_bytes_upper(self) -> int:
        """Declared bytes of retained JAX arrays, not device/compiler measurement.

        These bytes participate in conservative admission through ``bound``.
        Their real device buffers and compiler allocations remain unmanaged.
        """
        return self._logical_unmanaged_bytes_upper

    @property
    def retained_owner_objects(self) -> Mapping[int, object]:
        """Read the actual union of owned and live covered graph identities."""
        # ChainMap mutates only its first, owned dictionary. Borrowed layers
        # are read-only despite the constructor's MutableMapping annotation.
        return ChainMap(
            self._retained_owners,
            *(
                cast(MutableMapping[int, object], owners)
                for owners in self._shared_owners
            ),
            *(
                cast(MutableMapping[int, object], owners)
                for owners in self._covered_owners
            ),
        )

    @property
    def has_covered_retained_storage(self) -> bool:
        """Whether the caller currently holds explicit external graph coverage."""
        return bool(self._covered_owners)

    @contextmanager
    def cover_retained_storage(self, owners: Mapping[int, object], /) -> Iterator[None]:
        """Borrow graph coverage while its external live reservation is held."""
        if not self._active or self._adopted_by is not None:
            raise RuntimeError("Host storage coverage requires its active owner.")
        self._covered_owners.append(owners)
        try:
            yield
        finally:
            if self._covered_owners[-1] is not owners:
                raise RuntimeError("Host storage coverage must leave in nesting order.")
            self._covered_owners.pop()

    def adopt_retained_storage(
        self,
        ticket: BatchTicket,
        contributor: ManagedBytesContributor,
        owners: Mapping[int, object],
        logical_bytes: int,
        /,
    ) -> None:
        """Adopt the exact live ticket; never release or readmit its capacity."""
        if isinstance(ticket, NativeScopeTicket):
            self._adopt_scope_storage(ticket, contributor, owners, logical_bytes)
            return
        if (
            not self._active
            or self._adopted_by is not None
            or not isinstance(ticket, NativeBatchTicket)
            or ticket.scope is not self._budget
            or ticket.settled
            or contributor.ticket is not ticket
            or any(prior is ticket for prior, _ in self._adopted_reservations)
        ):
            raise RuntimeError(
                "Host storage adoption requires one live same-scope owner."
            )
        logical_total = self._logical_unmanaged_bytes_upper + _unsigned_budget(
            logical_bytes, "logical_bytes"
        )
        # Prepare both indices before committing; failures leave both owners
        # unchanged and the caller still holds the original live reservation.
        adopted = [*self._adopted_reservations, (ticket, contributor)]
        shared = [*self._shared_owners, owners]
        self._adopted_reservations = adopted
        self._shared_owners = shared
        self._logical_unmanaged_bytes_upper = logical_total

    def _adopt_scope_storage(
        self,
        ticket: NativeScopeTicket,
        contributor: ManagedBytesContributor,
        owners: Mapping[int, object],
        logical_bytes: int,
        /,
    ) -> None:
        """Transfer a storage-only scope view's live tokens, without readmission."""
        if (
            not self._active
            or self._adopted_by is not None
            or ticket.scope is not self._budget
            or ticket.settled
            or ticket._entered
            or contributor.ticket is not ticket
            or ticket._default_bytes
            or len(ticket._components) != 1
            or contributor not in ticket._components
            or ticket not in ticket.scope._ticket_views
        ):
            raise RuntimeError(
                "Host storage adoption requires one live storage-only scope view."
            )
        ticket._active_scope()
        prior_bound = self.bound
        count = len(ticket._workspaces)
        # Child-reference tuples, replacement lists, weak-owner/index slots and
        # rollback records are concrete new metadata, not a copy of their payload.
        self.set_bound(prior_bound + 256 * (16 + 12 * count))
        try:
            children = tuple(ticket._workspaces.values())
            for child in children:
                ancestor = child._budget
                while ancestor is not None and ancestor is not self._budget:
                    ancestor = ancestor.parent
                if (
                    ancestor is None
                    or child._library is not self._library
                    or not child._active
                    or child._adopted_by is not None
                    or child._token is not None
                    or child._adopted_workspaces
                    or child._adopted_reservations
                ):
                    raise RuntimeError(
                        "Scope storage adoption requires unchanged same-pool tokens."
                    )
            adopted = [*self._adopted_workspaces, *children]
            shared = [*self._shared_owners, owners]
            logical_total = self._logical_unmanaged_bytes_upper + _unsigned_budget(
                logical_bytes, "logical_bytes"
            )
        except BaseException:
            self.set_bound(prior_bound)
            raise
        # All potentially allocating work is complete. Commit the one closer,
        # then detach the old view without closing any live contribution.
        self._adopted_workspaces = adopted
        self._shared_owners = shared
        self._logical_unmanaged_bytes_upper = logical_total
        for child in children:
            child._adopted_by = self
        ticket._workspaces.clear()
        ticket._components.clear()
        ticket._default_bytes = 0
        ticket.settled = True
        del ticket.scope._ticket_views[ticket]

    def adopt_workspace(self, child: NativeHostStorageWorkspace, /) -> None:
        """Transfer an already admitted token and its retained graph unchanged."""
        if (
            not self._active
            or self._adopted_by is not None
            or child is self
            or not child._active
            or child._adopted_by is not None
            or child._budget is not self._budget
            or child._token is not None
            or child._adopted_workspaces
            or child._adopted_reservations
        ):
            raise RuntimeError("Host workspace adoption requires a separate live token.")
        logical_total = (
            self._logical_unmanaged_bytes_upper + child._logical_unmanaged_bytes_upper
        )
        adopted = [*self._adopted_workspaces, child]
        shared = [*self._shared_owners, child._retained_owners]
        self._adopted_workspaces = adopted
        self._shared_owners = shared
        self._logical_unmanaged_bytes_upper = logical_total
        child._adopted_by = self

    def forget_retained_owners_after_handover(self) -> None:
        """Drop scientific owner indices after an ended weak publication handover.

        The caller prepays this fixed owner-index pass while live and installs
        its weak root/array leases first. Tokens, bounds and allocator ancestry
        are unchanged: this is neither a release nor a new admission.
        """
        owners = (self, *self._adopted_workspaces)
        if self._adopted_by is not None or any(
            not owner._active
            or owner._budget._thread is not None
            or owner._budget.evidence is None
            or owner._token is not None
            or owner._covered_owners
            or owner._adopted_reservations
            or owner is not self and owner._adopted_workspaces
            for owner in owners
        ):
            raise RuntimeError("Weak handover requires ended, unborrowed workspace tokens.")
        for owner in owners:
            owner._retained_owners.clear()
            owner._shared_owners.clear()

    def retain_owner(
        self,
        value: object,
        /,
        *,
        existing: Mapping[int, object] | None = None,
        replace_bound: bool = False,
    ) -> None:
        """Admit a retained graph union, optionally completing a scratch phase."""
        if not self._active:
            raise RuntimeError(
                f"_meshcore.NativeHostStorageWorkspace.retain_owner.value must satisfy the input contract (Source retention requires its active host workspace.); got {_input_summary(value)}."
            )
        scope = current_native_execution_scope()
        if scope is None or scope._library is not self._library:
            raise RuntimeError(
                f"_meshcore.NativeHostStorageWorkspace.retain_owner.value must satisfy the input contract (Source retention requires its active native pool.); got {_input_summary(value)}."
            )

        if replace_bound and (
            self._retained_owners
            or self._adopted_workspaces
            or self._adopted_reservations
        ):
            raise RuntimeError("Host phase completion requires an unassigned token.")
        covered = self.retained_owner_objects
        if existing is not None:
            covered = ChainMap(
                cast(MutableMapping[int, object], covered),
                cast(MutableMapping[int, object], existing),
            )
        if not self._covered_owners and not replace_bound:
            # Other routes retain their existing contract; they have not
            # opted into the bounded-host publication/ownership proof.
            retained, declared, logical, _ = retained_host_storage_profile(
                value, covered, is_native_host_buffer
            )
            self.set_bound(self.bound + declared)
            self._retained_owners.update(retained)
            self._logical_unmanaged_bytes_upper += logical
            return
        with NativeHostStorageWorkspace(scope) as traversal:
            traversal.set_bound(1024)
            traversal_bound = 1024

            def admit_owner(owner: object) -> None:
                nonlocal traversal_bound
                # Pending edge tuples/list slots and visited-owner indices.
                # Special canonical records have <=8 children; collections
                # and dataclasses declare larger fanout before materialization.
                slots = 8
                if isinstance(owner, Mapping):
                    slots = max(slots, 2 * len(owner))
                elif isinstance(owner, (tuple, list, set, frozenset)):
                    slots = max(slots, len(owner))
                fields = getattr(type(owner), "__dataclass_fields__", None)
                if fields is not None:
                    slots = max(slots, len(fields))
                if isinstance(owner, np.ndarray) and owner.dtype.hasobject:
                    slots = max(slots, owner.size)
                if isinstance(owner, PyTreeDef):
                    slots = max(slots, owner.num_nodes + 2)
                traversal_bound += 512 + 512 * slots
                traversal.set_bound(traversal_bound)

            retained, declared, logical, _ = retained_host_storage_profile(
                value, covered, is_native_host_buffer, admit_owner=admit_owner
            )
            self.set_bound(declared if replace_bound else self.bound + declared)
            self._retained_owners.update(retained)
            self._logical_unmanaged_bytes_upper += logical

    def retain_source_metadata(self, source: object, /) -> weakref.finalize:
        """Atomically hand prepaid metadata to a weak completed-source lifetime.

        Only a metadata-only token admitted by the actual request root is
        eligible. All graph inspection, work, and capacity admission must have
        finished before root end. The returned finalizer captures neither this
        workspace nor its scope and permits early release by the publication
        owner; failed binding leaves the original reservation intact.
        """
        if (
            not self._source_metadata
            or not self._active
            or not self._budget._source_metadata_completed
            or self._budget._thread is not None
            or self._budget.parent is not None
            or current_native_execution_scope() is not None
            or self._token is not None
            or self._adopted_by is not None
            or self._retained_owners
            or self._covered_owners
            or self._shared_owners
            or self._adopted_reservations
            or self._adopted_workspaces
            or self._finalizer is None
        ):
            raise RuntimeError(
                "Source metadata retention requires its completed metadata-only root lease."
            )
        prior = self._finalizer
        invocation = prior.peek()
        if invocation is None:
            raise RuntimeError("Source metadata retention requires its live reservation.")
        _, callback, args, kwargs = invocation
        empty = ctypes.c_void_p()
        retained = weakref.finalize(source, callback, *args, **kwargs)
        retained.atexit = False
        # Both finalizers cover the same token until the old one is detached:
        # there is no release/readmission gap, nor an unowned native handle.
        prior.detach()
        self._finalizer = None
        self._reservation = empty
        self._active = False
        self.bound = 0
        self._source_metadata = False
        return retained

    def close(self) -> None:
        if self._adopted_by is not None:
            raise RuntimeError("Transferred host storage must close through its owner.")
        reservations, self._adopted_reservations = self._adopted_reservations, []
        children, self._adopted_workspaces = self._adopted_workspaces, []
        primary: BaseException | None = None
        for child in reversed(children):
            child._adopted_by = None
            try:
                child.close()
            except BaseException as error:
                if primary is None:
                    primary = error
        for ticket, contributor in reversed(reservations):
            try:
                if not ticket.settled:
                    contributor.close()
                    ticket.settle()
            except BaseException as error:
                if primary is None:
                    primary = error
        if self._finalizer is None and self._reservation.value is not None:
            self._budget._managed_observation_mutation(
                self._managed_observation_ticket,
                origin=self._managed_observation_origin,
            )
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        elif self._library is not None and self._reservation.value is not None:
            self._library["phx_mc_execution_release_host_storage"](self._reservation)
        self._reservation = ctypes.c_void_p()
        self._active = False
        self._retained_owners.clear()
        self._shared_owners.clear()
        self.bound = 0
        self._logical_unmanaged_bytes_upper = 0
        if self._token is not None:
            _active_host_workspace.reset(self._token)
            self._token = None
        if primary is not None:
            raise primary

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


_active_execution_scope: ContextVar[NativeExecutionScope | None] = ContextVar(
    "native_execution_scope",
    default=None,
)
_active_host_workspace: ContextVar[NativeHostStorageWorkspace | None] = ContextVar(
    "native_host_storage_workspace",
    default=None,
)


def current_native_host_workspace() -> NativeHostStorageWorkspace | None:
    """The innermost real host-storage owner of the active native pool."""
    workspace = _active_host_workspace.get()
    budget = current_native_execution_scope()
    if workspace is None or not workspace._active or budget is None:
        return None
    if workspace._library is not budget._library:
        raise RuntimeError(
            f"_meshcore.current_native_host_workspace.state must satisfy the input contract (A host workspace cannot move to another native allocation pool.); got {'refused lifecycle state'}."
        )
    return workspace


def current_native_execution_scope() -> NativeExecutionScope | None:
    """Return the ambient CPU-native owner without loading a library.

    A copied context cannot transfer a live native thread-local scope to another
    thread. Original request/operation-clock metadata remains Python-wrapper
    owned; this accessor exposes only the live native admission owner.
    """
    budget = _active_execution_scope.get()
    if budget is not None and (
        budget._thread != threading.get_ident() and budget._target_for_thread() is None
    ):
        raise RuntimeError(
            f"_meshcore.current_native_execution_scope.state must satisfy the input contract (A native execution budget belongs to its creating thread.); got {'refused lifecycle state'}."
        )
    return budget


@final
class PlanarExecutionEvidence(StrictModule):
    """Native measurements, including refused executions.

    ``work_evidence`` names work units, orientation/incircle/power requests,
    insertions, flips, peak cavity cells, work refusals and cavity refusals.
    ``memory_counters`` owns the six allocator measurements. Only the distinct
    nine-slot planar work array remains positional and readonly.
    """

    work_evidence: NDArray[np.uint64]
    memory_counters: NativeExecutionMemoryCounters
    status: MeshcoreStatus = eqx.field(static=True)

    def __post_init__(self) -> None:
        if not isinstance(self.status, MeshcoreStatus):
            raise TypeError(
                f"_meshcore.PlanarExecutionEvidence.__post_init__.self must be a MeshcoreStatus; got {_input_summary(self)}."
            )
        array = self.work_evidence
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.uint64):
            raise TypeError(
                f"PlanarExecutionEvidence.work_evidence must use a uint64 array; got {_input_summary(array)}."
            )
        if array.shape != (9,):
            raise ValueError(
                f"PlanarExecutionEvidence.work_evidence must have shape (9,); got {_input_summary(array)}."
            )
        array.setflags(write=False)
        if type(self.memory_counters) is not NativeExecutionMemoryCounters:
            raise TypeError(
                "PlanarExecutionEvidence.memory_counters must be NativeExecutionMemoryCounters; "
                f"got {_input_summary(self.memory_counters)}."
            )


@final
class NativeResourceFailure(MeshcoreError):
    """Native refusal preserving its declared work-counter identities and memory."""

    memory_counters: NativeExecutionMemoryCounters
    execution_evidence: NativeExecutionEvidence | None

    def __init__(
        self,
        evidence: PlanarExecutionEvidence | NativeExecutionEvidence,
        operation: str,
        /,
    ) -> None:
        super().__init__(
            evidence.status,
            f"{operation}: native execution refused",
            memory_counters=evidence.memory_counters,
        )
        match evidence:
            case PlanarExecutionEvidence():
                self.execution_evidence = None
                self.work_evidence = evidence.work_evidence
                self.work_evidence_kind = "planar"
            case NativeExecutionEvidence():
                self.execution_evidence = evidence
                self.memory_counters = evidence.memory_counters
                self.work_evidence = np.asarray(
                    (
                        *(value for _, value in evidence.work_counters.items()),
                        evidence.externally_charged_work,
                        evidence.externally_charged_geometry_queries,
                        evidence.native_primitive_queries,
                    ),
                    dtype=np.uint64,
                )
                self.work_evidence.setflags(write=False)
                self.work_evidence_kind = "execution_scope"
            case unreachable:
                assert_never(unreachable)


class IntersectionClass(IntEnum):
    """Contact class of a triangle pair or segment/triangle pair.

    ``DISJOINT``, ``SHARED_VERTEX`` and ``SHARED_EDGE`` are legal adjacency of a
    conforming complex; ``TOUCHING`` (any other contact with disjoint relative
    interiors), ``CROSSING`` (transversal contact through both interiors),
    ``COPLANAR_OVERLAP`` (coplanar contact of positive length or area) and
    ``COINCIDENT`` (identical triangles) are not.
    """

    DISJOINT = 0
    SHARED_VERTEX = 1
    SHARED_EDGE = 2
    TOUCHING = 3
    CROSSING = 4
    COPLANAR_OVERLAP = 5
    COINCIDENT = 6


_P = ctypes.c_void_p
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_U64 = ctypes.c_uint64
_F64 = ctypes.c_double


@final
class _ClipReferenceTetrahedronExactRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_clip_reference_tetrahedron_exact_request."""

    words: int | None
    word_count: int
    offsets: int | None
    signs: int | None
    scratch_limit: int
    work_limit: int
    plane_count: int
    vertex_capacity: int
    face_capacity: int
    entry_capacity: int
    supporting_planes: int | None
    incident_planes: int | None
    face_offsets: int | None
    face_planes: int | None
    face_vertices: int | None
    vertex_count: int | None
    face_count: int | None
    work_units: int | None

    _fields_ = [
        ("words", _P),
        ("word_count", ctypes.c_int64),
        ("offsets", _P),
        ("signs", _P),
        ("scratch_limit", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("plane_count", ctypes.c_int32),
        ("vertex_capacity", ctypes.c_int32),
        ("face_capacity", ctypes.c_int32),
        ("entry_capacity", ctypes.c_int32),
        ("supporting_planes", _P),
        ("incident_planes", _P),
        ("face_offsets", _P),
        ("face_planes", _P),
        ("face_vertices", _P),
        ("vertex_count", _P),
        ("face_count", _P),
        ("work_units", _P),
    ]


@final
class _ArrangeTetrahedraExactRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_arrange_tetrahedra_exact_request."""

    point_count: int
    plane_count: int
    tetrahedron_count: int
    job_count: int
    cut_count: int
    point_word_count: int
    plane_word_count: int
    point_words: int | None
    point_offsets: int | None
    point_signs: int | None
    plane_words: int | None
    plane_offsets: int | None
    plane_signs: int | None
    tetrahedra: int | None
    tetrahedron_planes: int | None
    parents: int | None
    cut_offsets: int | None
    cut_planes: int | None
    scratch_limit: int
    work_limit: int
    evidence: int | None
    result: int | None

    _fields_ = [
        ("point_count", ctypes.c_int64),
        ("plane_count", ctypes.c_int64),
        ("tetrahedron_count", ctypes.c_int64),
        ("job_count", ctypes.c_int64),
        ("cut_count", ctypes.c_int64),
        ("point_word_count", ctypes.c_int64),
        ("plane_word_count", ctypes.c_int64),
        ("point_words", _P),
        ("point_offsets", _P),
        ("point_signs", _P),
        ("plane_words", _P),
        ("plane_offsets", _P),
        ("plane_signs", _P),
        ("tetrahedra", _P),
        ("tetrahedron_planes", _P),
        ("parents", _P),
        ("cut_offsets", _P),
        ("cut_planes", _P),
        ("scratch_limit", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("evidence", _P),
        ("result", _P),
    ]

@final
class _ClipBoxHalfspacesRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_clip_box_halfspaces_request."""

    count: int
    box_lower: int | None
    box_upper: int | None
    plane_capacity: int
    normals: int | None
    offsets: int | None
    plane_counts: int | None
    vertex_capacity: int
    face_capacity: int
    face_vertex_capacity: int
    vertices: int | None
    vertex_counts: int | None
    face_offsets: int | None
    face_labels: int | None
    face_vertices: int | None
    face_counts: int | None
    volumes: int | None
    first_moments: int | None
    item_status: int | None

    _fields_ = [
        ("count", ctypes.c_int64),
        ("box_lower", _P),
        ("box_upper", _P),
        ("plane_capacity", ctypes.c_int32),
        ("normals", _P),
        ("offsets", _P),
        ("plane_counts", _P),
        ("vertex_capacity", ctypes.c_int32),
        ("face_capacity", ctypes.c_int32),
        ("face_vertex_capacity", ctypes.c_int32),
        ("vertices", _P),
        ("vertex_counts", _P),
        ("face_offsets", _P),
        ("face_labels", _P),
        ("face_vertices", _P),
        ("face_counts", _P),
        ("volumes", _P),
        ("first_moments", _P),
        ("item_status", _P),
    ]


@final
class _ConstrainedDelaunay2DRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_constrained_delaunay_2d_request."""

    point_count: int
    points: int | None
    segment_count: int
    segments: int | None
    hole_count: int
    holes: int | None
    keep_convex_hull: int
    min_angle_degrees: float
    max_area: float
    max_edge_length: float
    lattice_spacing: float
    max_steiner: int
    max_triangles: int
    max_cavity_cells: int
    max_work: int
    max_scratch_bytes: int
    work_evidence: int | None
    memory_evidence: int | None
    witness_capacity: int
    witness_parents: int | None
    witness_segments: int | None
    witness_ratios: int | None
    mesh: int | None

    _fields_ = [
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("segment_count", ctypes.c_int64),
        ("segments", _P),
        ("hole_count", ctypes.c_int64),
        ("holes", _P),
        ("keep_convex_hull", ctypes.c_int32),
        ("min_angle_degrees", ctypes.c_double),
        ("max_area", ctypes.c_double),
        ("max_edge_length", ctypes.c_double),
        ("lattice_spacing", ctypes.c_double),
        ("max_steiner", ctypes.c_int64),
        ("max_triangles", ctypes.c_int64),
        ("max_cavity_cells", ctypes.c_int64),
        ("max_work", ctypes.c_int64),
        ("max_scratch_bytes", ctypes.c_int64),
        ("work_evidence", _P),
        ("memory_evidence", _P),
        ("witness_capacity", ctypes.c_int64),
        ("witness_parents", _P),
        ("witness_segments", _P),
        ("witness_ratios", _P),
        ("mesh", _P),
    ]


@final
class _SurfaceReconnectRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_surface_reconnect_request."""

    vertex_count: int
    charts: int | None
    points: int | None
    normals: int | None
    vertex_metrics: int | None
    vertex_pole_ids: int | None
    triangle_count: int
    triangles: int | None
    constrained: int | None
    insert_count: int
    insert_charts: int | None
    insert_points: int | None
    insert_normals: int | None
    insert_metrics: int | None
    insert_pole_ids: int | None
    insert_spacing: int | None
    insert_hints: int | None
    insert_edges: int | None
    sweep: int
    max_triangles: int
    work_limit: int
    triangles_out: int | None
    constrained_out: int | None
    triangle_count_out: int | None
    vertex_ids: int | None
    item_status: int | None
    counters: int | None

    _fields_ = [
        ("vertex_count", ctypes.c_int64),
        ("charts", _P),
        ("points", _P),
        ("normals", _P),
        ("vertex_metrics", _P),
        ("vertex_pole_ids", _P),
        ("triangle_count", ctypes.c_int64),
        ("triangles", _P),
        ("constrained", _P),
        ("insert_count", ctypes.c_int64),
        ("insert_charts", _P),
        ("insert_points", _P),
        ("insert_normals", _P),
        ("insert_metrics", _P),
        ("insert_pole_ids", _P),
        ("insert_spacing", _P),
        ("insert_hints", _P),
        ("insert_edges", _P),
        ("sweep", ctypes.c_int32),
        ("max_triangles", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("triangles_out", _P),
        ("constrained_out", _P),
        ("triangle_count_out", _P),
        ("vertex_ids", _P),
        ("item_status", _P),
        ("counters", _P),
    ]


@final
class _TetMeshCreateRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_tet_mesh_create_request."""

    point_count: int
    points: int | None
    tet_count: int
    tets: int | None
    tet_regions: int | None
    face_count: int
    faces: int | None
    face_sources: int | None
    segment_count: int
    segments: int | None
    segment_sources: int | None
    protection_radii: int | None
    source_point_count: int
    source_points: int | None
    source_face_count: int
    source_faces: int | None
    source_face_group_rows: int | None
    source_face_tolerances: int | None
    source_segment_count: int
    source_segments: int | None
    source_segment_group_rows: int | None
    source_segment_tolerances: int | None
    witness_strata: int | None
    witness_entities: int | None
    witness_parameters: int | None
    curved_source_count: int
    curved_sources: int | None
    curved_tolerances: int | None
    curved_digests: int | None
    face_claims: int | None
    boundary_policy: int
    max_vertices: int
    max_tetrahedra: int
    max_scratch_bytes: int
    mesh: int | None

    _fields_ = [
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("tet_count", ctypes.c_int64),
        ("tets", _P),
        ("tet_regions", _P),
        ("face_count", ctypes.c_int64),
        ("faces", _P),
        ("face_sources", _P),
        ("segment_count", ctypes.c_int64),
        ("segments", _P),
        ("segment_sources", _P),
        ("protection_radii", _P),
        ("source_point_count", ctypes.c_int64),
        ("source_points", _P),
        ("source_face_count", ctypes.c_int64),
        ("source_faces", _P),
        ("source_face_group_rows", _P),
        ("source_face_tolerances", _P),
        ("source_segment_count", ctypes.c_int64),
        ("source_segments", _P),
        ("source_segment_group_rows", _P),
        ("source_segment_tolerances", _P),
        ("witness_strata", _P),
        ("witness_entities", _P),
        ("witness_parameters", _P),
        ("curved_source_count", ctypes.c_int64),
        ("curved_sources", _P),
        ("curved_tolerances", _P),
        ("curved_digests", _P),
        ("face_claims", _P),
        ("boundary_policy", ctypes.c_int32),
        ("max_vertices", ctypes.c_int64),
        ("max_tetrahedra", ctypes.c_int64),
        ("max_scratch_bytes", ctypes.c_int64),
        ("mesh", _P),
    ]


@final
class _Plc3DRecoverRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_plc3d_recover_request."""

    point_count: int
    points: int | None
    polygon_count: int
    polygon_offsets: int | None
    polygon_vertices: int | None
    polygon_facets: int | None
    facet_count: int
    facet_regions: int | None
    segment_count: int
    segments: int | None
    seed_count: int
    seeds: int | None
    seed_regions: int | None
    boundary_policy: int
    facet_tolerances: int | None
    segment_tolerances: int | None
    max_vertices: int
    max_tetrahedra: int
    work_limit: int
    maximum_scratch_bytes: int
    measure_phases: int
    result: int | None

    _fields_ = [
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("polygon_count", ctypes.c_int64),
        ("polygon_offsets", _P),
        ("polygon_vertices", _P),
        ("polygon_facets", _P),
        ("facet_count", ctypes.c_int64),
        ("facet_regions", _P),
        ("segment_count", ctypes.c_int64),
        ("segments", _P),
        ("seed_count", ctypes.c_int64),
        ("seeds", _P),
        ("seed_regions", _P),
        ("boundary_policy", ctypes.c_int32),
        ("facet_tolerances", _P),
        ("segment_tolerances", _P),
        ("max_vertices", ctypes.c_int64),
        ("max_tetrahedra", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("maximum_scratch_bytes", ctypes.c_uint64),
        ("measure_phases", ctypes.c_int32),
        ("result", _P),
    ]


@final
class _RestrictedPowerCellsRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_restricted_power_cells_request."""

    site_count: int
    sites: int | None
    weights: int | None
    neighbor_offsets: int | None
    neighbors: int | None
    split_sites: int | None
    point_count: int
    points: int | None
    tet_count: int
    tets: int | None
    tet_regions: int | None
    tet_face_facets: int | None
    max_pieces: int
    max_vertices: int
    work_limit: int
    max_edges: int
    max_faces: int
    max_face_entries: int
    max_cells: int
    max_site_witnesses: int
    max_sided_loop_occurrences: int
    max_sided_corner_occurrences: int
    max_connectivity_entries: int
    max_data_bytes: int
    record_phases: int
    result: int | None

    _fields_ = [
        ("site_count", ctypes.c_int64),
        ("sites", _P),
        ("weights", _P),
        ("neighbor_offsets", _P),
        ("neighbors", _P),
        ("split_sites", _P),
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("tet_count", ctypes.c_int64),
        ("tets", _P),
        ("tet_regions", _P),
        ("tet_face_facets", _P),
        ("max_pieces", ctypes.c_int64),
        ("max_vertices", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("max_edges", ctypes.c_int64),
        ("max_faces", ctypes.c_int64),
        ("max_face_entries", ctypes.c_int64),
        ("max_cells", ctypes.c_int64),
        ("max_site_witnesses", ctypes.c_int64),
        ("max_sided_loop_occurrences", ctypes.c_int64),
        ("max_sided_corner_occurrences", ctypes.c_int64),
        ("max_connectivity_entries", ctypes.c_int64),
        ("max_data_bytes", ctypes.c_int64),
        ("record_phases", ctypes.c_int8),
        ("result", _P),
    ]


@final
class _RestrictedPowerCellsExactRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_restricted_power_cells_exact_request."""

    original_site_count: int
    original_sites: int | None
    original_weights: int | None
    image_count: int
    image_site_owners: int | None
    image_coordinate_offsets: int | None
    component_count: int
    image_coordinate_components: int | None
    neighbor_offsets: int | None
    neighbors: int | None
    split_sites: int | None
    point_count: int
    points: int | None
    tet_count: int
    tets: int | None
    tet_regions: int | None
    tet_face_facets: int | None
    max_pieces: int
    max_vertices: int
    work_limit: int
    max_edges: int
    max_faces: int
    max_face_entries: int
    max_cells: int
    max_site_witnesses: int
    max_sided_loop_occurrences: int
    max_sided_corner_occurrences: int
    max_connectivity_entries: int
    max_data_bytes: int
    record_phases: int
    result: int | None

    _fields_ = [
        ("original_site_count", ctypes.c_int64),
        ("original_sites", _P),
        ("original_weights", _P),
        ("image_count", ctypes.c_int64),
        ("image_site_owners", _P),
        ("image_coordinate_offsets", _P),
        ("component_count", ctypes.c_int64),
        ("image_coordinate_components", _P),
        ("neighbor_offsets", _P),
        ("neighbors", _P),
        ("split_sites", _P),
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("tet_count", ctypes.c_int64),
        ("tets", _P),
        ("tet_regions", _P),
        ("tet_face_facets", _P),
        ("max_pieces", ctypes.c_int64),
        ("max_vertices", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("max_edges", ctypes.c_int64),
        ("max_faces", ctypes.c_int64),
        ("max_face_entries", ctypes.c_int64),
        ("max_cells", ctypes.c_int64),
        ("max_site_witnesses", ctypes.c_int64),
        ("max_sided_loop_occurrences", ctypes.c_int64),
        ("max_sided_corner_occurrences", ctypes.c_int64),
        ("max_connectivity_entries", ctypes.c_int64),
        ("max_data_bytes", ctypes.c_int64),
        ("record_phases", ctypes.c_int8),
        ("result", _P),
    ]


@final
class _PolyhedronCornerHexesRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_polyhedron_corner_hexes_request."""

    nv: int
    ne: int
    nf: int
    nc: int
    face_entries: int
    cell_face_entries: int
    corner_entries: int
    edges: int | None
    face_offsets: int | None
    face_vertices: int | None
    cell_offsets: int | None
    cell_faces: int | None
    vertex_offsets: int | None
    cell_vertices: int | None
    corner_orientations: int | None
    vertex_count: int
    hex_capacity: int
    scratch_limit: int
    work_limit: int
    hexes: int | None
    parent_cells: int | None
    hex_count: int | None
    witness: int | None
    work_units: int | None

    _fields_ = [
        ("nv", ctypes.c_int64),
        ("ne", ctypes.c_int64),
        ("nf", ctypes.c_int64),
        ("nc", ctypes.c_int64),
        ("face_entries", ctypes.c_int64),
        ("cell_face_entries", ctypes.c_int64),
        ("corner_entries", ctypes.c_int64),
        ("edges", _P),
        ("face_offsets", _P),
        ("face_vertices", _P),
        ("cell_offsets", _P),
        ("cell_faces", _P),
        ("vertex_offsets", _P),
        ("cell_vertices", _P),
        ("corner_orientations", _P),
        ("vertex_count", ctypes.c_int64),
        ("hex_capacity", ctypes.c_int64),
        ("scratch_limit", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("hexes", _P),
        ("parent_cells", _P),
        ("hex_count", _P),
        ("witness", _P),
        ("work_units", _P),
    ]


@final
class _LevelSetSplit3DRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_level_set_split_3d_request."""

    point_count: int
    points: int | None
    values: int | None
    cell_count: int
    cells: int | None
    face_count: int
    faces: int | None
    edge_count: int
    edges: int | None
    protected_edges: int | None
    max_vertices: int
    max_cells: int
    work_limit: int
    cavity_limit: int
    output_mode: int
    out_points: int | None
    out_cells: int | None
    out_parents: int | None
    out_sources: int | None
    out_weights: int | None
    out_deviations: int | None
    counters: int | None

    _fields_ = [
        ("point_count", ctypes.c_int64),
        ("points", _P),
        ("values", _P),
        ("cell_count", ctypes.c_int64),
        ("cells", _P),
        ("face_count", ctypes.c_int64),
        ("faces", _P),
        ("edge_count", ctypes.c_int64),
        ("edges", _P),
        ("protected_edges", _P),
        ("max_vertices", ctypes.c_int64),
        ("max_cells", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("cavity_limit", ctypes.c_int64),
        ("output_mode", ctypes.c_int32),
        ("out_points", _P),
        ("out_cells", _P),
        ("out_parents", _P),
        ("out_sources", _P),
        ("out_weights", _P),
        ("out_deviations", _P),
        ("counters", _P),
    ]


@final
class _SurfaceEnvelopeCreateRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_surface_envelope_create_request."""

    vertices: int | None
    vertex_count: int
    faces: int | None
    face_count: int
    offset: float
    spacing: float
    certificate_spacing: float
    sample_limit: int
    work_limit: int
    surface_vertex_capacity: int
    surface_triangle_capacity: int
    volume_vertex_capacity: int
    tetrahedron_capacity: int
    result: int | None
    counts: int | None
    bounds: int | None

    _fields_ = [
        ("vertices", _P),
        ("vertex_count", ctypes.c_int64),
        ("faces", _P),
        ("face_count", ctypes.c_int64),
        ("offset", ctypes.c_double),
        ("spacing", ctypes.c_double),
        ("certificate_spacing", ctypes.c_double),
        ("sample_limit", ctypes.c_int64),
        ("work_limit", ctypes.c_int64),
        ("surface_vertex_capacity", ctypes.c_int64),
        ("surface_triangle_capacity", ctypes.c_int64),
        ("volume_vertex_capacity", ctypes.c_int64),
        ("tetrahedron_capacity", ctypes.c_int64),
        ("result", _P),
        ("counts", _P),
        ("bounds", _P),
    ]


@final
class _TetMeshCommitEdgeInsertionRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_tet_mesh_commit_edge_insertion_request."""

    mesh: int | None
    first: int
    second: int
    split_position: int | None
    final_position: int | None
    source_fraction: float
    maximum_cavity_cells: int
    insertion_kind: int
    accepted_generation: int
    removed_count: int
    removed_tetrahedra: int | None
    proposed_count: int
    proposed_tetrahedra: int | None
    target_size: float
    work_limit: int
    inserted_vertex: int | None
    projected_source: int
    claim_count: int
    claim_digest: int | None
    claim_faces: int | None
    claim_coordinates: int | None
    claim_deviations: int | None

    _fields_ = [
        ("mesh", _P),
        ("first", ctypes.c_int32),
        ("second", ctypes.c_int32),
        ("split_position", _P),
        ("final_position", _P),
        ("source_fraction", ctypes.c_double),
        ("maximum_cavity_cells", ctypes.c_int64),
        ("insertion_kind", ctypes.c_int32),
        ("accepted_generation", ctypes.c_int64),
        ("removed_count", ctypes.c_int64),
        ("removed_tetrahedra", _P),
        ("proposed_count", ctypes.c_int64),
        ("proposed_tetrahedra", _P),
        ("target_size", ctypes.c_double),
        ("work_limit", ctypes.c_int64),
        ("inserted_vertex", _P),
        ("projected_source", _I32),
        ("claim_count", _I64),
        ("claim_digest", _P),
        ("claim_faces", _P),
        ("claim_coordinates", _P),
        ("claim_deviations", _P),
    ]


@final
class _TetMeshCurvedCollapseRequest(ctypes.Structure):
    """Naturally aligned transport for phx_mc_tet_mesh_curved_collapse_request."""

    mesh: int | None
    remove: int
    keep: int
    source: int
    accepted_generation: int
    digest: int | None
    claim_count: int
    claim_faces: int | None
    claim_coordinates: int | None
    claim_deviations: int | None
    require_improvement: int
    work_limit: int
    applied: int | None

    _fields_ = [
        ("mesh", _P),
        ("remove", ctypes.c_int32),
        ("keep", ctypes.c_int32),
        ("source", ctypes.c_int32),
        ("accepted_generation", ctypes.c_int64),
        ("digest", _P),
        ("claim_count", ctypes.c_int64),
        ("claim_faces", _P),
        ("claim_coordinates", _P),
        ("claim_deviations", _P),
        ("require_improvement", ctypes.c_int32),
        ("work_limit", ctypes.c_int64),
        ("applied", _P),
    ]


@final
class _PeriodicRefinementInput(ctypes.Structure):
    """Naturally aligned transport for phx_mc_periodic_refinement_input."""

    point_count: int
    points: int | None
    lattice: int | None
    inverse: int | None
    cell_count: int
    vertices: int | None
    shifts: int | None
    control_count: int
    controls: int | None
    statistic_count: int
    quantiles: int | None
    absolute_tolerance: float
    relative_tolerance: float
    max_vertices: int
    max_cells: int
    max_images: int
    max_work_units: int
    max_cavity_cells: int
    max_scratch_bytes: int
    protected_edge_count: int
    protected_edges: int | None

    _fields_ = [
        ("point_count", _I64),
        ("points", _P),
        ("lattice", _P),
        ("inverse", _P),
        ("cell_count", _I64),
        ("vertices", _P),
        ("shifts", _P),
        ("control_count", _I64),
        ("controls", _P),
        ("statistic_count", _I64),
        ("quantiles", _P),
        ("absolute_tolerance", _F64),
        ("relative_tolerance", _F64),
        ("max_vertices", _I64),
        ("max_cells", _I64),
        ("max_images", _I64),
        ("max_work_units", _I64),
        ("max_cavity_cells", _I64),
        ("max_scratch_bytes", _U64),
        ("protected_edge_count", _I64),
        ("protected_edges", _P),
    ]


_SIGNATURES: dict[str, tuple[object, tuple[object, ...]]] = {
    "phx_mc_version": (ctypes.c_char_p, ()),
    "phx_mc_abi_contract": (ctypes.c_char_p, ()),
    "phx_mc_build_hash": (ctypes.c_char_p, ()),
    "phx_mc_build_configuration": (ctypes.c_char_p, ()),
    "phx_mc_exact_domain": (None, (_P, _P)),
    "phx_mc_parallel_begin": (_I32, (_I32, _F64, _P)),
    "phx_mc_parallel_end": (_I32, (_P,)),
    "phx_mc_execution_begin": (_I32, (_U64, _U64, _U64, _U64, _F64, _P, _P)),
    "phx_mc_execution_batch_admit": (_I32, (_P, _P, _U64, _U64, _U64, _U64, _P)),
    "phx_mc_execution_batch_settle": (_I32, (_P, _U64, _U64, _U64, _U64)),
    "phx_mc_execution_batch_checkpoint": (_I32, (_P,)),
    "phx_mc_execution_batch_refusal": (_I32, (_P, _P)),
    "phx_mc_execution_batch_usage": (_I32, (_P, _P)),
    "phx_mc_execution_target_begin": (_I32, (_P, _P, _P)),
    "phx_mc_execution_target_end": (_I32, (_P,)),
    "phx_mc_execution_batch_refuse": (_I32, (_P, _P, _P, _P)),
    "phx_mc_execution_refusal_request": (_I32, (_P, _P)),
    "phx_mc_execution_allocator_bookkeeping": (_I32, (_P, _P)),
    "phx_mc_execution_end": (_I32, (_P, _P, _P, _P)),
    "phx_mc_execution_charge": (_I32, (_P, _U64, _U64)),
    "phx_mc_execution_import_preparation": (_I32, (_P, _U64, _U64, _F64)),
    "phx_mc_execution_prior_seconds": (_I32, (_P, _P)),
    "phx_mc_execution_admit_work_bound": (_I32, (_P, _U64)),
    "phx_mc_execution_admit_cavity": (_I32, (_P, _U64)),
    "phx_mc_execution_remaining": (_I32, (_P, _P, _P)),
    "phx_mc_execution_allocate_host_array": (_I32, (_P, _U64, _U64, _P, _P)),
    "phx_mc_execution_free_host_array": (None, (_P,)),
    "phx_mc_execution_reserve_host_storage": (_I32, (_P, _U64, _P)),
    "phx_mc_execution_release_host_storage": (None, (_P,)),
    "phx_mc_orient2d": (_I32, (_I64, _P, _P, _P, _P)),
    "phx_mc_orient3d": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_incircle": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_insphere": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_orient2d_sos": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_orient3d_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_incircle_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_insphere_sos": (_I32, (_I64, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_polygon_intersection_moments": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_moments": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P),
    ),
    "phx_mc_polygon_intersection_simplices": (
        _I32,
        (_I64, _I32, _P, _P, _I32, _P, _P, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedron_intersection_simplices": (
        _I32,
        (_I64, _P, _P, _I32, _I32, _P, _P, _P, _P, _P),
    ),
    "phx_mc_polyhedron_clip_moments": (
        _I32,
        (_I64, _I32, _P, _P, _P, _P, _I32, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfplanes": (
        _I32,
        (_I64, _P, _P, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_clip_reference_triangle_exact": (
        _I32,
        (_P, _I64, _P, _P, _I64, _I64, _I32, _P, _P, _P, _P),
    ),
    "phx_mc_clip_reference_tetrahedron_exact": (
        _I32,
        (ctypes.POINTER(_ClipReferenceTetrahedronExactRequest),),
    ),
    "phx_mc_arrange_tetrahedra_exact": (
        _I32,
        (ctypes.POINTER(_ArrangeTetrahedraExactRequest),),
    ),
    "phx_mc_tetrahedra_arrangement_sizes": (_I32, (_P, _P)),
    "phx_mc_tetrahedra_arrangement_export": (
        _I32,
        (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tetrahedra_arrangement_free": (None, (_P,)),
    "phx_mc_clip_box_halfspaces": (_I32, (ctypes.POINTER(_ClipBoxHalfspacesRequest),)),
    "phx_mc_delaunay_2d": (_I32, (_I64, _P, _I64, _I64, _I64, _I64, _P, _P, _P)),
    "phx_mc_regular_2d": (_I32, (_I64, _P, _P, _I64, _I64, _I64, _I64, _P, _P, _P)),
    "phx_mc_delaunay_3d": (_I32, (_I64, _P, _I64, _P)),
    "phx_mc_regular_3d": (_I32, (_I64, _P, _P, _I64, _P)),
    "phx_mc_triangle_intersections": (
        _I32,
        (_I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_segment_triangle_intersections": (
        _I32,
        (_I64, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_point_triangle_locations": (_I32, (_I64, _P, _P, _P, _P, _P)),
    "phx_mc_simplex_quadratic_bounds": (
        _I32,
        (_I64, _I32, _I32, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_quadratic_tetrahedron_validity": (
        _I32,
        (_I64, _P, _P, _F64, _I64, _I64, _I64, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_interval_program_execute": (
        _I32,
        (_P, _I64, _P, _I64, _P, _I64, _P, _P, _I64),
    ),
    "phx_mc_constrained_delaunay_2d": (
        _I32,
        (ctypes.POINTER(_ConstrainedDelaunay2DRequest),),
    ),
    "phx_mc_mesh_dimension": (_I32, (_P,)),
    "phx_mc_mesh_point_count": (_I64, (_P,)),
    "phx_mc_mesh_cell_count": (_I64, (_P,)),
    "phx_mc_mesh_input_point_count": (_I64, (_P,)),
    "phx_mc_mesh_copy_points": (_I32, (_P, _P)),
    "phx_mc_mesh_copy_cells": (_I32, (_P, _P)),
    "phx_mc_mesh_copy_vertex_map": (_I32, (_P, _P)),
    "phx_mc_mesh_copy_cell_constraints": (_I32, (_P, _P)),
    "phx_mc_mesh_copy_cell_regions": (_I32, (_P, _P)),
    "phx_mc_mesh_free": (None, (_P,)),
    "phx_mc_triangulation_3d_create": (_I32, (_I64, _P, _I64, _I64, _I64, _P)),
    "phx_mc_triangulation_3d_insert": (_I32, (_P, _I64, _P, _I64, _P, _P)),
    "phx_mc_triangulation_3d_constrain_facets": (_I32, (_P, _I64, _P, _P, _P)),
    "phx_mc_triangulation_3d_label_regions": (_I32, (_P, _I64, _P, _P, _P)),
    "phx_mc_triangulation_3d_locate": (_I32, (_P, _I64, _P, _P, _P, _P)),
    "phx_mc_triangulation_3d_statistics": (_I32, (_P, _P)),
    "phx_mc_triangulation_3d_finalize": (_I32, (_P, _P)),
    "phx_mc_triangulation_3d_free": (None, (_P,)),
    "phx_mc_graph_partition": (
        _I32,
        (_I64, _P, _P, _P, _P, _I32, _P, _P, _I32, _I32, _I64, _P, _P),
    ),
    "phx_mc_surface_reconnect": (_I32, (ctypes.POINTER(_SurfaceReconnectRequest),)),
    "phx_mc_periodic_delaunay": (
        _I32,
        (_I32, _I64, _P, _P, _P, _P, _F64, _I64, _I64, _P, _P),
    ),
    "phx_mc_periodic_cell_count": (_I64, (_P,)),
    "phx_mc_periodic_copy_cells": (_I32, (_P, _P, _P)),
    "phx_mc_periodic_free": (None, (_P,)),
    "phx_mc_periodic_refine_2d": (
        _I32,
        (ctypes.POINTER(_PeriodicRefinementInput), _P, _P),
    ),
    "phx_mc_periodic_refinement_point_count": (_I64, (_P,)),
    "phx_mc_periodic_refinement_cell_count": (_I64, (_P,)),
    "phx_mc_periodic_refinement_copy_points": (_I32, (_P, _P)),
    "phx_mc_periodic_refinement_copy_cells": (_I32, (_P, _P, _P)),
    "phx_mc_periodic_refinement_free": (None, (_P,)),
    "phx_mc_plc3d_recover": (_I32, (ctypes.POINTER(_Plc3DRecoverRequest),)),
    "phx_mc_plc3d_source_constraints": (
        _I32,
        (_I64, _P, _I64, _P, _P, _P, _I64, _P, _I64, _P, _I64, _U64, _I32, _P),
    ),
    "phx_mc_plc3d_sizes": (_I32, (_P, _P)),
    "phx_mc_plc3d_counters": (_I32, (_P, _P)),
    "phx_mc_plc3d_failure": (_I32, (_P, _P)),
    "phx_mc_plc3d_export": (_I32, (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_plc3d_source_witnesses": (_I32, (_P, _P, _P, _P, _P, _P)),
    "phx_mc_plc3d_free": (None, (_P,)),
    "phx_mc_plc3d_phase_times": (_I32, (_P, _P, _P, _P)),
    "phx_mc_plc3d_memory_evidence": (_I32, (_P, _P, _P)),
    "phx_mc_tet_mesh_create": (_I32, (ctypes.POINTER(_TetMeshCreateRequest),)),
    "phx_mc_tet_mesh_refine": (_I32, (_P, _P, _F64, _I64, _I64, _P)),
    "phx_mc_tet_mesh_improve": (_I32, (_P, _F64, _F64, _I32, _I64, _P)),
    "phx_mc_tet_mesh_flip_face": (_I32, (_P, _P, _I32, _P)),
    "phx_mc_tet_mesh_remove_edge": (_I32, (_P, _I32, _I32, _I32, _P)),
    "phx_mc_tet_mesh_relocate": (_I32, (_P, _I32, _P, _I32, _P)),
    "phx_mc_tet_mesh_relocate_vertices": (_I32, (_P, _I64, _P, _P, _F64, _F64, _I64, _P)),
    "phx_mc_tet_mesh_remove_vertex": (_I32, (_P, _I32, _I32, _P)),
    "phx_mc_tet_mesh_quality": (_I32, (_P, _F64, _P, _P, _P)),
    "phx_mc_tet_mesh_proposal_shape": (
        _I32,
        (_P, _I64, _P, _P, _I64, _P, _F64, _I64, _P),
    ),
    "phx_mc_tet_mesh_counts": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_work_units": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_set_work_limit": (_I32, (_P, _I64)),
    "phx_mc_tet_mesh_export": (_I32, (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_unmet": (_I32, (_P, _P, _P, _P)),
    "phx_mc_tet_mesh_free": (None, (_P,)),
    "phx_mc_restricted_power_cells": (
        _I32,
        (ctypes.POINTER(_RestrictedPowerCellsRequest),),
    ),
    "phx_mc_restricted_power_cells_exact": (
        _I32,
        (ctypes.POINTER(_RestrictedPowerCellsExactRequest),),
    ),
    "phx_mc_power_cells_sizes": (_I32, (_P, _P)),
    "phx_mc_power_cells_preflight": (_I32, (_P, _P)),
    "phx_mc_power_cells_capacity": (_I32, (_P, _P)),
    "phx_mc_power_cells_counters": (_I32, (_P, _P)),
    "phx_mc_power_cells_source_work_units": (_I64, (_P,)),
    "phx_mc_power_cells_failure": (_I32, (_P, _P)),
    "phx_mc_power_cells_export": (
        _I32,
        (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_power_cells_free": (None, (_P,)),
    "phx_mc_power_cells_export_vertex_carriers": (_I32, (_P, _P)),
    "phx_mc_power_cells_site_witness_size": (_I64, (_P,)),
    "phx_mc_power_cells_export_vertex_sites": (_I32, (_P, _P, _P)),
    "phx_mc_power_cells_phase_seconds": (_I32, (_P, _P)),
    "phx_mc_arrange_triangles": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _P, _I64, _I64, _P, _P),
    ),
    "phx_mc_arrangement_sizes": (_I32, (_P, _P)),
    "phx_mc_arrangement_counters": (_I32, (_P, _P)),
    "phx_mc_arrangement_export": (_I32, (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_arrangement_free": (None, (_P,)),
    "phx_mc_arrangement_classify": (_I32, (_P, _I64, _I64, _P, _P)),
    "phx_mc_arrangement_export_classification": (_I32, (_P, _P, _P)),
    "phx_mc_triangle_dual_quads": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_tetrahedron_dual_hexes": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_polyhedron_corner_hexes": (
        _I32,
        (ctypes.POINTER(_PolyhedronCornerHexesRequest),),
    ),
    "phx_mc_surface_envelope_create": (
        _I32,
        (ctypes.POINTER(_SurfaceEnvelopeCreateRequest),),
    ),
    "phx_mc_surface_envelope_arrays": (_I32, (_P, _P, _P, _P, _P)),
    "phx_mc_surface_envelope_free": (None, (_P,)),
    "phx_mc_level_set_split_3d": (_I32, (ctypes.POINTER(_LevelSetSplit3DRequest),)),
    "phx_mc_restricted_dual_3d": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _I64, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_mixed_template_counts": (_I32, (_I32, _I32, _I32, _P)),
    "phx_mc_mixed_template": (_I32, (_I32, _I32, _I32, _I64, _P)),
    "phx_mc_restricted_centers_3d": (_I32, (_I64, _P, _I64, _P, _P, _P)),
    "phx_mc_tet_mesh_split_edge": (_I32, (_P, _I32, _I32, _P, _F64, _F64, _I64, _P)),
    "phx_mc_tet_mesh_split_edge_relocate": (
        _I32,
        (_P, _I32, _I32, _P, _P, _F64, _I64, _P),
    ),
    "phx_mc_tet_mesh_inspect_edge_insertion": (
        _I32,
        (_P, _I32, _I32, _P, _P, _F64, _I64, _I64, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_inspect_projected_edge_insertion": (
        _I32,
        (_P, _I32, _I32, _P, _P, _F64, _I32, _I64, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_commit_edge_insertion": (
        _I32,
        (ctypes.POINTER(_TetMeshCommitEdgeInsertionRequest),),
    ),
    "phx_mc_tet_mesh_collapse_edge": (_I32, (_P, _I32, _I32, _I32, _I64, _P)),
    "phx_mc_tet_mesh_inspect_curved_collapse": (
        _I32,
        (_P, _I32, _I32, _I32, _I64, _I64, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_curved_collapse_cavity": (
        _I32,
        (_P, _I32, _I32, _I64, _I64, _I64, _I64, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_collapse_curved_edge": (
        _I32,
        (ctypes.POINTER(_TetMeshCurvedCollapseRequest),),
    ),
    "phx_mc_tet_mesh_accepted_generation": (_I32, (_P, _P)),
    "phx_mc_tet_mesh_curved_claims": (_I32, (_P, _I64, _P, _P, _P, _P, _P, _P)),
    "phx_mc_tet_mesh_remove_multiface": (_I32, (_P, _I64, _P, _I32, _I32, _I64, _P)),
    "phx_mc_tet_mesh_exude": (_I32, (_P, _F64, _F64, _F64, _F64, _I32, _I64, _P, _P)),
    "phx_mc_restricted_rays_3d": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_reconnect": (_I32, (_P, _I64, _P, _I64, _P, _I64, _P)),
    "phx_mc_tet_mesh_protect_vertices": (_I32, (_P, _I64, _P)),
    "phx_mc_tet_mesh_construct_edge_split": (
        _I32,
        (_P, _I32, _I32, _F64, _I64, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_construct_curve_points": (
        _I32,
        (_P, _I64, _P, _P, _P, _I64, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_source_evidence": (
        _I32,
        (_P, _I64, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_tet_mesh_source_refusals": (_I32, (_P, _I64, _P, _P, _P, _P, _P)),
    "phx_mc_subdivide_hex_grid": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_tet_mesh_measure_execution": (_I32, (_P, _I32)),
    "phx_mc_tet_mesh_execution_times": (_I32, (_P, _P, _P)),
    "phx_mc_tet_mesh_set_memory_limit": (_I32, (_P, _I64)),
    "phx_mc_tet_mesh_memory_evidence": (_I32, (_P, _P)),
}


def _library_location(configured: str, /) -> Path | str:
    """Return the library path, or the reason it is unavailable."""

    if configured:
        path = Path(configured)
        if not path.is_file():
            return f"{LIBRARY_ENVIRONMENT['meshcore']} names a missing meshcore library: {configured!r}."
        return path
    if importlib.util.find_spec("phydrax_meshcore") is None:
        return (
            "The meshcore library is unavailable: install the optional "
            "'phydrax[meshcore]' extra or set "
            f"{LIBRARY_ENVIRONMENT['meshcore']} to libphydrax_meshcore."
        )
    return Path(importlib.import_module("phydrax_meshcore").library_path())


_MANAGED_NATIVE_CONTROL = frozenset(
    (
        "phx_mc_version",
        "phx_mc_abi_contract",
        "phx_mc_build_hash",
        "phx_mc_build_configuration",
        "phx_mc_exact_domain",
        "phx_mc_parallel_begin",
        "phx_mc_parallel_end",
        "phx_mc_execution_begin",
        "phx_mc_execution_end",
        "phx_mc_execution_allocator_bookkeeping",
        "phx_mc_execution_charge",
        "phx_mc_execution_batch_admit",
        "phx_mc_execution_batch_settle",
        "phx_mc_execution_batch_checkpoint",
        "phx_mc_execution_batch_usage",
        "phx_mc_execution_target_begin",
        "phx_mc_execution_target_end",
        "phx_mc_execution_batch_refuse",
        "phx_mc_execution_batch_refusal",
        "phx_mc_execution_refusal_request",
        "phx_mc_execution_import_preparation",
        "phx_mc_execution_prior_seconds",
        "phx_mc_execution_admit_work_bound",
        "phx_mc_execution_admit_cavity",
        "phx_mc_execution_remaining",
    )
)
_MANAGED_NATIVE_RETAINED = frozenset(
    (
        "phx_mc_execution_allocate_host_array",
        "phx_mc_execution_reserve_host_storage",
    )
)
_MANAGED_NATIVE_FREE = frozenset(
    (
        "phx_mc_execution_free_host_array",
        "phx_mc_execution_release_host_storage",
        "phx_mc_mesh_free",
        "phx_mc_triangulation_3d_free",
        "phx_mc_periodic_free",
        "phx_mc_periodic_refinement_free",
        "phx_mc_tet_mesh_free",
        "phx_mc_plc3d_free",
        "phx_mc_power_cells_free",
        "phx_mc_arrangement_free",
        "phx_mc_tetrahedra_arrangement_free",
        "phx_mc_surface_envelope_free",
    )
)


def _observe_managed_native_call(
    result: Any, function: Any, arguments: tuple[Any, ...], /
) -> Any:
    """Observe actual typed entries without replacing their C callable identity.

    This ABI has no Python callbacks: a lexical native target is still installed
    when ctypes performs its post-call check. Never transform a return status or
    retain its arguments; the owning Python wrapper still performs all cleanup.
    """
    scope = current_native_execution_scope()
    if scope is not None:
        scope._managed_native_entry(getattr(function, "__name__", ""), function)
    return result


@final
class MeshcoreLibrary:
    """Loaded meshcore shared library with typed C entry points."""

    __slots__ = (
        "_functions",
        "_library",
        "build_hash",
        "binary_hash",
        "configuration",
        "path",
        "version",
    )

    def __init__(
        self,
        library: ctypes.CDLL,
        functions: Mapping[str, Callable[..., object]],
        path: Path,
        version: str,
        build_hash: str,
        binary_hash: str,
        configuration: str,
        /,
    ) -> None:
        self._library = library
        self._functions = functions
        for function in functions.values():
            if (
                isinstance(function, CFuncPtr)
                and getattr(function, "errcheck", None) is None
            ):
                function.errcheck = _observe_managed_native_call
        self.path = path
        self.version = version
        self.build_hash = build_hash
        self.binary_hash = binary_hash
        self.configuration = configuration

    def __getitem__(self, name: str, /) -> Any:
        return self._functions[name]


def _identity_text(function: Callable[[], object], symbol: str, path: Path, /) -> Any:
    """Decode one required non-null ASCII identity string, or return its error."""

    raw = function()
    if not isinstance(raw, bytes):
        return None, f"meshcore library {str(path)!r} returned null from {symbol}."
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError:
        return None, f"meshcore library {str(path)!r} returned non-ASCII {symbol}."
    if not value:
        return None, f"meshcore library {str(path)!r} returned an empty {symbol}."
    return value, None


def _bind(library: ctypes.CDLL, path: Path, binary_hash: str, /) -> MeshcoreLibrary | str:
    """Verify the C ABI contract, bind every entry point and the release, or return the reason."""

    # No other entry point is typed or called before the declarations match.
    try:
        contract_function = library["phx_mc_abi_contract"]
    except AttributeError:
        return (
            f"meshcore library {str(path)!r} exports no phx_mc_abi_contract, so its C ABI "
            f"cannot be verified against {_ABI_CONTRACT}; install the phydrax-meshcore "
            "build of this phydrax."
        )
    restype, argtypes = _SIGNATURES["phx_mc_abi_contract"]
    # ty: ignore[invalid-assignment]
    contract_function.restype = restype
    # ty: ignore[invalid-assignment]
    contract_function.argtypes = argtypes
    contract, error = _identity_text(contract_function, "phx_mc_abi_contract", path)
    if error is not None:
        return error
    if contract != _ABI_CONTRACT:
        return (
            f"meshcore library {str(path)!r} implements C ABI contract {contract}, but this "
            f"phydrax binds {_ABI_CONTRACT}; install the phydrax-meshcore build of this phydrax."
        )
    functions = {}
    missing = []
    for name, (restype, argtypes) in _SIGNATURES.items():
        # ctypes reports an unresolved symbol only as AttributeError.
        try:
            function = library[name]
        except AttributeError:
            missing.append(name)
            continue
        # ty: ignore[invalid-assignment]
        function.restype = restype
        # ty: ignore[invalid-assignment]
        function.argtypes = argtypes
        functions[name] = function
    if missing:
        return (
            f"meshcore library {str(path)!r} lacks C ABI symbols "
            f"{', '.join(missing)}; install the phydrax-meshcore release of this phydrax."
        )
    version, error = _identity_text(functions["phx_mc_version"], "phx_mc_version", path)
    if error is not None:
        return error
    build_hash, error = _identity_text(
        functions["phx_mc_build_hash"], "phx_mc_build_hash", path
    )
    if error is not None:
        return error
    if len(build_hash) != 64 or any(
        character not in "0123456789abcdef" for character in build_hash
    ):
        return f"meshcore library {str(path)!r} returned an invalid phx_mc_build_hash."
    configuration, error = _identity_text(
        functions["phx_mc_build_configuration"], "phx_mc_build_configuration", path
    )
    if error is not None:
        return error
    clock_capabilities = tuple(
        item for item in configuration.split("|") if item.startswith("execution_clocks=")
    )
    if clock_capabilities != (_EXECUTION_CLOCK_CAPABILITY,):
        return (
            f"meshcore library {str(path)!r} does not declare the required "
            f"{_EXECUTION_CLOCK_CAPABILITY!r} capability exactly once; install the "
            "matched phydrax-meshcore build for independent request and workflow clocks."
        )
    try:
        required = importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError:
        return (
            "The meshcore release cannot be verified: phydrax distribution "
            "metadata is not installed."
        )
    if version != required:
        return (
            f"meshcore library {str(path)!r} is release {version}, but phydrax "
            f"{required} requires phydrax-meshcore=={required}."
        )
    return MeshcoreLibrary(
        library, functions, path, version, build_hash, binary_hash, configuration
    )


@functools.cache
def _resolve(configured: str, /) -> MeshcoreLibrary | str:
    location = _library_location(configured)
    if isinstance(location, str):
        return location
    # Capture the loaded artifact once, not whatever a later build places at
    # the same path. A concurrent replacement invalidates this load attempt.
    try:
        before = location.stat()
        with location.open("rb") as stream:
            binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        library = ctypes.CDLL(str(location))
        after = location.stat()
    except OSError as error:
        return f"meshcore library {str(location)!r} cannot be loaded: {error}"
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        return f"meshcore library {str(location)!r} changed while loading."
    return _bind(library, location, binary_hash)


def _current() -> MeshcoreLibrary | str:
    return _resolve((library_input("meshcore") or "").strip())


def meshcore_available() -> bool:
    """Whether the native meshcore library can be loaded."""

    return not isinstance(_current(), str)


@functools.cache
def _provider_installed(configured: str, /) -> bool:
    return bool(configured) or importlib.util.find_spec("phydrax_meshcore") is not None


def meshcore_provider_installed() -> bool:
    """Whether a meshcore provider is configured or installed, loadable or not.

    True when the library environment names a path (even a missing one) or the
    ``phydrax_meshcore`` package is importable. Unlike :func:`meshcore_available`,
    an ABI or load failure of a present provider still counts as installed, so
    callers route to :func:`load_meshcore` and its error instead of a fallback.
    """

    return _provider_installed((library_input("meshcore") or "").strip())


def load_meshcore() -> MeshcoreLibrary:
    """Return the loaded meshcore library or raise :class:`MeshcoreUnavailableError`."""

    resolved = _current()
    if isinstance(resolved, str):
        raise MeshcoreUnavailableError(resolved)
    return resolved


def meshcore_identity() -> str:
    """Library release and source build hash: ``phydrax-meshcore <release> <sha256>``."""

    library = load_meshcore()
    return f"phydrax-meshcore {library.version} {library.build_hash}"


def meshcore_runtime_identity() -> dict[str, str]:
    """Loaded source/binary/configuration identity, excluding timings."""
    library = load_meshcore()
    return {
        "release": library.version,
        "source_digest": library.build_hash,
        "binary_digest": library.binary_hash,
        "configuration": library.configuration,
    }


def meshcore_exact_domain() -> tuple[int, int]:
    """Binary exponent range ``(minimum, maximum)`` of exactly handled coordinates.

    Coordinates must be zero or have magnitude in ``[2**minimum, 2**maximum]``;
    weights use the doubled range.
    """

    library = load_meshcore()
    minimum = np.zeros((1,), dtype=np.int32)
    maximum = np.zeros((1,), dtype=np.int32)
    library["phx_mc_exact_domain"](minimum.ctypes.data, maximum.ctypes.data)
    return int(minimum[0]), int(maximum[0])


# ---------------------------------------------------------------- validation


def _real_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(
            f"{'_meshcore.' + name} must be a real floating or integer array; got {_input_summary(value)}."
        )
    if np.issubdtype(array.dtype, np.integer) and np.any(
        (array > 2**53) | (array < -(2**53))
    ):
        raise ValueError(
            f"{'_meshcore.' + name} must be exactly representable; got {_input_summary(value)}."
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _index_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"{'_meshcore.' + name} must be an integer array; got {_input_summary(value)}."
        )
    return array


def _points(value: object, name: str, width: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n, {width}); got {_input_summary(value)}."
        )
    return array


def _batched_points(
    arrays: tuple[object, ...], names: tuple[str, ...], width: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    converted = tuple(
        _real_array(value, name) for value, name in zip(arrays, names, strict=True)
    )
    for array, name in zip(converted, names, strict=True):
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(
                f"{'_meshcore.' + name} must have shape (..., {width}); got {_input_summary(width)}."
            )
    leading = np.broadcast_shapes(*(array.shape[:-1] for array in converted))
    flat = tuple(
        np.ascontiguousarray(
            np.broadcast_to(array, leading + (width,)).reshape((-1, width)),
            dtype=np.float64,
        )
        for array in converted
    )
    return flat, leading


def _raise_for_call(status: int, operation: str, /) -> None:
    code = MeshcoreStatus(status)
    match code:
        case MeshcoreStatus.OK:
            return
        case MeshcoreStatus.NONFINITE_INPUT:
            raise ValueError(
                f"{operation + '.inputs'} must be finite; got {_input_summary(status)}."
            )
        case MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                f"{operation + '.coordinates'} must be zero or have magnitude in "
                f"[2**-120, 2**120] (weights [2**-240, 2**240]); got native status {code.name}."
            )
        case MeshcoreStatus.DEGENERATE_INPUT:
            raise ValueError(
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: the distinct points do not span the space.); got {_input_summary(status)}."
            )
        case MeshcoreStatus.INVALID_ARGUMENT | MeshcoreStatus.INVALID_INPUT:
            raise ValueError(
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: invalid input ({code.name}).); got {_input_summary(status)}."
            )
        case MeshcoreStatus.CONSTRAINT_INTERSECTION:
            raise ValueError(
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: constraint segments cross in their interiors.); got {_input_summary(status)}."
            )
        case MeshcoreStatus.CAPACITY_EXCEEDED:
            raise MeshcoreError(
                code,
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: explicit capacity exceeded); got {_input_summary(status)}.",
            )
        case MeshcoreStatus.TIMEOUT:
            raise MeshcoreError(
                code,
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: native wall-clock deadline reached); got {_input_summary(status)}.",
            )
        case MeshcoreStatus.REFINEMENT_LIMIT | MeshcoreStatus.INTERNAL_ERROR:
            raise MeshcoreError(
                code,
                f"{operation + '.inputs'} must satisfy the input contract ({operation}: native failure); got {_input_summary(status)}.",
            )
        case _:
            raise ValueError(
                f"{operation + '.inputs'} must satisfy the input contract (Unknown meshcore status {status}.); got {_input_summary(status)}."
            )


# ---------------------------------------------------------------- predicates


def _predicate(
    name: str, arrays: tuple[object, ...], width: int, /, ids: object = None
) -> np.ndarray:
    """Keep predicate scratch on the live request without charging a host tariff.

    Validation precedes target admission. The scope view reuses a mapped target
    or, only while observing, prepays a phase from the actual remaining allowance.
    """
    library = load_meshcore()
    names = tuple(f"point {index}" for index in range(len(arrays)))
    flat, leading = _batched_points(arrays, names, width)
    count = flat[0].shape[0]
    signs = np.zeros((count,), dtype=np.int8)
    pointers = [array.ctypes.data for array in flat]
    if ids is not None:
        index = _index_array(ids, "ids")
        if index.shape[-1:] != (len(arrays),):
            raise ValueError(
                f"_meshcore._predicate.ids must have shape (..., {len(arrays)}); got {_input_summary(ids)}."
            )
        index_flat = np.ascontiguousarray(
            np.broadcast_to(index, leading + (len(arrays),)).reshape((-1, len(arrays))),
            dtype=np.int64,
        )
        pointers.append(index_flat.ctypes.data)
    scope = current_native_execution_scope()
    with ExitStack() as stack:
        if scope is not None:
            ticket = stack.enter_context(scope.ticket_view())
            stack.enter_context(ticket.native_target())
        status = library[name](count, *pointers, signs.ctypes.data)
        _raise_for_call(status, name.removeprefix("phx_mc_"))
    return signs.reshape(leading)


def exact_orient2d(a: object, b: object, c: object, /) -> np.ndarray:
    """``sign det[b - a, c - a]`` for ``(..., 2)`` arrays (int8 in {-1, 0, 1})."""

    return _predicate("phx_mc_orient2d", (a, b, c), 2)


def exact_orient3d(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """``sign det[b - a, c - a, d - a]`` for ``(..., 3)`` arrays."""

    return _predicate("phx_mc_orient3d", (a, b, c, d), 3)


def exact_incircle(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """In-circle sign: positive when ``d`` is inside the circle of CCW ``(a, b, c)``."""

    return _predicate("phx_mc_incircle", (a, b, c, d), 2)


def exact_insphere(
    a: object, b: object, c: object, d: object, e: object, /
) -> np.ndarray:
    """In-sphere sign: positive when ``e`` is inside the sphere of positive ``(a..d)``."""

    return _predicate("phx_mc_insphere", (a, b, c, d, e), 3)


def symbolic_orient2d(a: object, b: object, c: object, ids: object, /) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient2d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient2d_sos", (a, b, c), 2, ids)


def symbolic_orient3d(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Index-ordered Simulation-of-Simplicity ``orient3d``; never zero for distinct ids."""

    return _predicate("phx_mc_orient3d_sos", (a, b, c, d), 3, ids)


def symbolic_incircle(
    a: object, b: object, c: object, d: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``incircle``; zero only for four collinear points."""

    return _predicate("phx_mc_incircle_sos", (a, b, c, d), 2, ids)


def symbolic_insphere(
    a: object, b: object, c: object, d: object, e: object, ids: object, /
) -> np.ndarray:
    """Lifted-perturbation ``insphere``; zero only for five coplanar points."""

    return _predicate("phx_mc_insphere_sos", (a, b, c, d, e), 3, ids)


# ------------------------------------------------- quadratic simplex bounds


def simplex_quadratic_bounds(
    q_lower: object,
    q_upper: object,
    barycentric_lower: object,
    barycentric_upper: object,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Outward enclosures of degree-two Bernstein forms on child simplices.

    ``q_lower``/``q_upper`` ``(n, c, m, m)`` enclose symmetric matrices ``Q`` of
    ``f(lambda) = lambda^T Q lambda`` over root barycentric coordinates (``Q_ij``,
    ``i != j``, is the coefficient of ``2 lambda_i lambda_j``); a lower endpoint
    may be ``-inf`` and an upper endpoint ``+inf``. ``barycentric_lower``/
    ``barycentric_upper`` ``(n, m, m)`` enclose, row ``k``, the root barycentric
    coordinates of child vertex ``k``. Returns ``(lower, upper)`` ``(n, c)``
    enclosing ``f`` on each whole child for every enclosed ``Q`` and map; an
    infinite endpoint is inconclusive on its side only. ``c`` in ``{1, 2, 3}``
    and ``m`` in ``{2, 3, 4}`` are read only as the array layout.
    """
    library = load_meshcore()
    names = ("q_lower", "q_upper", "barycentric_lower", "barycentric_upper")
    values = (q_lower, q_upper, barycentric_lower, barycentric_upper)
    lower_q, upper_q, lower_rows, upper_rows = (
        _real_array(value, f"simplex_quadratic_bounds.{name}")
        for value, name in zip(values, names, strict=True)
    )
    if (
        lower_q.ndim != 4
        or lower_q.shape[1] not in (1, 2, 3)
        or lower_q.shape[2] not in (2, 3, 4)
        or lower_q.shape[3] != lower_q.shape[2]
    ):
        raise ValueError(
            f"_meshcore.simplex_quadratic_bounds.q_lower must have shape (n, c, m, m) with c in {{1, 2, 3}} and m in {{2, 3, 4}}; got {_input_summary(q_lower)}."
        )
    if upper_q.shape != lower_q.shape:
        raise ValueError(
            f"_meshcore.simplex_quadratic_bounds.q_upper must have shape {lower_q.shape}; got {_input_summary(q_upper)}."
        )
    count, components, vertices = lower_q.shape[:3]
    for rows, name, value in (
        (lower_rows, "barycentric_lower", barycentric_lower),
        (upper_rows, "barycentric_upper", barycentric_upper),
    ):
        if rows.shape != (count, vertices, vertices):
            raise ValueError(
                f"_meshcore.simplex_quadratic_bounds.{name} must have shape {(count, vertices, vertices)}; got {_input_summary(value)}."
            )
    scope = current_native_execution_scope()
    shape = (count, components)
    lower, upper = (
        (np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.float64))
        if scope is None
        else (
            scope.allocate_host_array(shape, np.float64),
            scope.allocate_host_array(shape, np.float64),
        )
    )
    with ExitStack() as stack:
        if scope is not None:
            ticket = stack.enter_context(scope.ticket_view())
            stack.enter_context(ticket.native_target())
        status = library["phx_mc_simplex_quadratic_bounds"](
            count,
            vertices - 1,
            components,
            lower_q.ctypes.data,
            upper_q.ctypes.data,
            lower_rows.ctypes.data,
            upper_rows.ctypes.data,
            lower.ctypes.data,
            upper.ctypes.data,
        )
        _raise_for_call(status, "simplex_quadratic_bounds")
    return lower, upper


def quadratic_tetrahedron_validity(
    nodes: NDArray[np.float64],
    derivative_weights: NDArray[np.integer],
    relative_floor: float,
    maximum_bernstein_nodes: int,
    maximum_subdivision_depth: int,
    maximum_work_units: int,
    maximum_scratch_bytes: int,
    /,
) -> QuadraticTetrahedronValidityResult:
    """Certify fresh P2 tetrahedron roots with the exact native flat ABI.

    ``nodes`` is ``(count, 10, 3)`` binary64 and ``derivative_weights`` is
    ``(3, 4, 10)`` integer power coefficients in constant, x, y, z order.
    The returned evidence is ``(work, queries, peak bytes, completed, failing
    cell, failure category)``. Native staging is admitted before any input copy.
    """
    operation = "quadratic_tetrahedron_validity"
    if not isinstance(nodes, np.ndarray) or nodes.dtype != np.dtype(np.float64):
        raise TypeError(
            f"_meshcore.{operation}.nodes must be a float64 NumPy array; got {_input_summary(nodes)}."
        )
    if nodes.ndim != 3 or nodes.shape[1:] != (10, 3):
        raise ValueError(
            f"_meshcore.{operation}.nodes must have shape (count, 10, 3); got {_input_summary(nodes)}."
        )
    count = nodes.shape[0]
    int64_max = (1 << 63) - 1
    if count > int64_max // 30 or count * 30 * 8 > np.iinfo(np.intp).max:
        raise ValueError(
            f"_meshcore.{operation}.nodes must fit addressable int64 counts and binary64 storage; got {_input_summary(nodes)}."
        )
    if not isinstance(derivative_weights, np.ndarray):
        raise TypeError(
            f"_meshcore.{operation}.derivative_weights must be an integer NumPy array; got {_input_summary(derivative_weights)}."
        )
    if (
        derivative_weights.shape != (3, 4, 10)
        or derivative_weights.dtype.hasobject
        or not np.issubdtype(derivative_weights.dtype, np.integer)
        or np.issubdtype(derivative_weights.dtype, np.bool_)
    ):
        raise ValueError(
            f"_meshcore.{operation}.derivative_weights must be an integer array with shape (3, 4, 10); got {_input_summary(derivative_weights)}."
        )
    if isinstance(relative_floor, (bool, np.bool_)) or not isinstance(
        relative_floor, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"_meshcore.{operation}.relative_floor must be a finite binary64 scalar; got {_input_summary(relative_floor)}."
        )
    floor = float(relative_floor)
    if not math.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError(
            f"_meshcore.{operation}.relative_floor must lie in [0, 1); got {_input_summary(relative_floor)}."
        )
    node_limit = bounded_integer(
        maximum_bernstein_nodes,
        f"_meshcore.{operation}.maximum_bernstein_nodes",
        lower=1,
        upper=int64_max,
    )
    depth_limit = bounded_integer(
        maximum_subdivision_depth,
        f"_meshcore.{operation}.maximum_subdivision_depth",
        lower=0,
        upper=int64_max,
    )
    work_limit = bounded_integer(
        maximum_work_units,
        f"_meshcore.{operation}.maximum_work_units",
        lower=0,
        upper=int64_max,
    )
    scratch_limit = bounded_integer(
        maximum_scratch_bytes,
        f"_meshcore.{operation}.maximum_scratch_bytes",
        lower=0,
        upper=int64_max,
    )
    for value in derivative_weights.flat:
        integer = int(value)
        if not -(1 << 31) <= integer < (1 << 31):
            raise ValueError(
                f"_meshcore.{operation}.derivative_weights must fit int32 coefficients; got {_input_summary(derivative_weights)}."
            )

    input_bytes = count * 30 * 8 + 3 * 4 * 10 * 4
    output_bytes = count * (4 + 8 + 8) + 6 * 8
    staging_upper = input_bytes + output_bytes + 6 * 256
    if staging_upper > int64_max or scratch_limit > _MAX_COUNTER_VALUE - staging_upper:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"_meshcore.{operation}.inputs must fit the bounded native staging pool; got {_input_summary(nodes)}.",
        )
    library = load_meshcore()
    execution = NativeExecutionScope(
        NativeExecutionLimits(
            work_limit,
            count,
            0,
            staging_upper + scratch_limit,
            math.inf,
        )
    )
    with execution:
        staged_nodes = execution.allocate_host_array((count, 10, 3), np.float64)
        staged_weights = execution.allocate_host_array((3, 4, 10), np.int32)
        decisions = execution.allocate_host_array((count,), np.int32)
        lower = execution.allocate_host_array((count,), np.float64)
        upper = execution.allocate_host_array((count,), np.float64)
        evidence = execution.allocate_host_array((6,), np.int64)
        np.copyto(staged_nodes, nodes, casting="no")
        np.copyto(staged_weights, derivative_weights, casting="unsafe")
        allowance = execution.remaining()
        _raise_for_call(allowance.status, f"{operation}_allowance")
        effective_work = min(work_limit, allowance.remaining_work_units)
        effective_scratch = min(scratch_limit, allowance.remaining_managed_bytes)
        native_status = MeshcoreStatus(
            library["phx_mc_quadratic_tetrahedron_validity"](
                count,
                staged_nodes.ctypes.data,
                staged_weights.ctypes.data,
                floor,
                node_limit,
                depth_limit,
                effective_work,
                effective_scratch,
                decisions.ctypes.data,
                lower.ctypes.data,
                upper.ctypes.data,
                evidence.ctypes.data,
            )
        )
        if native_status is not MeshcoreStatus.OK:
            raise QuadraticTetrahedronValidityError(native_status, evidence)
        result = QuadraticTetrahedronValidityResult(decisions, lower, upper, evidence)
    return result


# ------------------------------------------------- interval program execution


class IntervalProgram(Protocol):
    """Straight-line interval program over one float64 lower/upper workspace.

    Structural view of ``phydrax.geometry._interval_native.NativeIntervalProgram``:
    ``instructions`` int64 ``(rows, 8)``, ``indices`` int64 ``(k,)``,
    ``parameters`` float64 ``(p,)`` and ``template_lower``/``template_upper``
    float64 ``(workspace,)``, all C-contiguous; the input boxes occupy
    ``input_size`` entries at ``input_offset`` and the result
    ``prod(output_shape)`` entries at ``output_offset``.
    """

    @property
    def instructions(self) -> np.ndarray: ...
    @property
    def indices(self) -> np.ndarray: ...
    @property
    def parameters(self) -> np.ndarray: ...
    @property
    def template_lower(self) -> np.ndarray: ...
    @property
    def template_upper(self) -> np.ndarray: ...
    @property
    def input_offset(self) -> int: ...
    @property
    def input_size(self) -> int: ...
    @property
    def output_offset(self) -> int: ...
    @property
    def output_shape(self) -> tuple[int, ...]: ...


def _program_array(
    value: object, dtype: type[np.int64] | type[np.float64], ndim: int, name: str, /
) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(dtype)
        or value.ndim != ndim
        or (ndim == 2 and value.shape[1] != 8)
        or not value.flags.c_contiguous
    ):
        layout = "(rows, 8)" if ndim == 2 else "(n,)"
        raise ValueError(
            f"_meshcore.interval_program_execute.program.{name} must be a C-contiguous "
            f"{np.dtype(dtype).name} array of shape {layout}; got {_input_summary(value)}."
        )
    return value


def _program_extent(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 0:
        raise ValueError(
            f"_meshcore.interval_program_execute.program.{name} must be a nonnegative integer; got {_input_summary(value)}."
        )
    return int(value)


def interval_program_execute(
    program: IntervalProgram, lower: object, upper: object, /
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute ``program`` on input boxes ``[lower, upper]``.

    ``lower``/``upper`` hold ``program.input_size`` float64 values in C order,
    either in one array or an ordered tuple of float64 array blocks. Blocks
    copy directly into the admitted workspace without an intermediate packed
    allocation. All inputs are validated before that allocation; the kernel
    validates every instruction before its first write. Returns bounds as views
    of that private per-call workspace. Any native refusal raises, so a
    workspace left partial by a deadline refusal is discarded, never returned.
    An active execution scope owns and admits the workspace through its native
    host allocator. Returned views retain that allocation after scope exit.
    The kernel charges no execution work; it polls the original deadline and
    recorded refusals on the current paid target.
    """
    library = load_meshcore()
    instructions = _program_array(program.instructions, np.int64, 2, "instructions")
    indices = _program_array(program.indices, np.int64, 1, "indices")
    parameters = _program_array(program.parameters, np.float64, 1, "parameters")
    template_lower = _program_array(program.template_lower, np.float64, 1, "template_lower")
    template_upper = _program_array(program.template_upper, np.float64, 1, "template_upper")
    workspace = template_lower.shape[0]
    if template_upper.shape != template_lower.shape:
        raise ValueError(
            f"_meshcore.interval_program_execute.program.template_upper must have shape {template_lower.shape}; got {_input_summary(template_upper)}."
        )
    input_offset = _program_extent(program.input_offset, "input_offset")
    input_size = _program_extent(program.input_size, "input_size")
    output_offset = _program_extent(program.output_offset, "output_offset")
    output_shape = program.output_shape
    if not isinstance(output_shape, tuple):
        raise ValueError(
            f"_meshcore.interval_program_execute.program.output_shape must be a tuple of nonnegative integers; got {_input_summary(output_shape)}."
        )
    output_size = math.prod(
        _program_extent(extent, "output_shape") for extent in output_shape
    )
    if input_offset + input_size > workspace or output_offset + output_size > workspace:
        raise ValueError(
            f"_meshcore.interval_program_execute.program must place its input and output inside the {workspace}-entry workspace; got input [{input_offset}, {input_offset + input_size}) and output [{output_offset}, {output_offset + output_size})."
        )
    boxes: list[tuple[np.ndarray, ...]] = []
    for value, name in ((lower, "lower"), (upper, "upper")):
        if isinstance(value, tuple) and value and all(
            isinstance(part, np.ndarray) and part.dtype == np.float64 for part in value
        ):
            # The owning validation above establishes every block's kind and
            # dtype; the checker cannot propagate the all(...) predicate.
            arrays = cast(tuple[np.ndarray, ...], value)
        else:
            arrays = (np.asarray(value),)
        if any(array.dtype != np.float64 for array in arrays) or sum(
            array.size for array in arrays
        ) != input_size:
            raise ValueError(
                f"_meshcore.interval_program_execute.{name} must be a float64 array of {input_size} values; got {_input_summary(value)}."
            )
        boxes.append(arrays)
    scope = current_native_execution_scope()
    with ExitStack() as stack:
        if scope is None:
            buffers = np.empty((2, workspace), dtype=np.float64)
        else:
            ticket = stack.enter_context(scope.ticket_view())
            stack.enter_context(ticket.native_target())
            buffers = scope.allocate_host_array((2, workspace), np.float64)
        workspace_lower, workspace_upper = buffers
        workspace_lower[:] = template_lower
        workspace_upper[:] = template_upper
        for destination, arrays in zip((workspace_lower, workspace_upper), boxes, strict=True):
            offset = input_offset
            for array in arrays:
                stop = offset + array.size
                destination[offset:stop].reshape(array.shape)[...] = array
                offset = stop
        status = library["phx_mc_interval_program_execute"](
            instructions.ctypes.data,
            instructions.shape[0],
            indices.ctypes.data,
            indices.shape[0],
            parameters.ctypes.data,
            parameters.shape[0],
            workspace_lower.ctypes.data,
            workspace_upper.ctypes.data,
            workspace,
        )
        _raise_for_call(status, "interval_program_execute")
        output = slice(output_offset, output_offset + output_size)
        return (
            workspace_lower[output].reshape(output_shape),
            workspace_upper[output].reshape(output_shape),
        )


# ---------------------------------------------------------------- clipping


def _counted_polygons(
    vertices: object, counts: object, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    array = _real_array(vertices, name)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n, k, 2); got {_input_summary(vertices)}."
        )
    count_array = _index_array(counts, f"{name} counts")
    if count_array.shape != array.shape[:1]:
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n,); got {_input_summary(counts)}."
        )
    if np.any(count_array < 0) or np.any(count_array > array.shape[1]):
        raise ValueError(
            f"{'_meshcore.' + name} must lie in [0, {array.shape[1]}]; got {_input_summary(counts)}."
        )
    return array, np.ascontiguousarray(count_array, dtype=np.int32)


def polygon_intersection_moments(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of convex polygon pairs.

    Polygons are ``(n, k, 2)`` padded vertex arrays with per-row counts, in either
    orientation. Returns ``(area (n,), first_moment (n, 2), status (n,))`` where
    ``first_moment`` is the integral of ``x`` over the intersection and ``status``
    holds :class:`MeshcoreStatus` item codes (empty and contact overlaps are ``OK``
    with zero area).
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"_meshcore.polygon_intersection_moments.batches must have equal length; got {first.shape[0]} and {second.shape[0]}."
        )
    count = first.shape[0]
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_moments"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_moments")
    return area, moment, status


def polygon_intersection_simplices(
    first_vertices: object,
    first_counts: object,
    second_vertices: object,
    second_counts: object,
    /,
    *,
    simplex_capacity: int | None = None,
) -> PolygonIntersectionSimplices:
    """Convex polygon-pair intersections as triangle partitions plus their moments.

    Returns :class:`PolygonIntersectionSimplices` with ``simplices (n, S, 3, 2), simplex_counts (n,), area, first_moment,
    status``. Each nonempty intersection is fanned from the vertex average of
    its counterclockwise loop, the fan of ``polygon_intersection_moments``, so the
    signed triangle areas sum to ``area`` term by term. The default capacity
    ``k1 + k2`` bounds every intersection.
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"_meshcore.polygon_intersection_simplices.batches must have equal length; got {first.shape[0]} and {second.shape[0]}."
        )
    capacity = _capacity(
        simplex_capacity, max(1, first.shape[1] + second.shape[1]), "simplex_capacity"
    )
    count = first.shape[0]
    simplices = np.zeros((count, capacity, 3, 2), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polygon_intersection_simplices"](
        count,
        first.shape[1],
        first.ctypes.data,
        first_count.ctypes.data,
        second.shape[1],
        second.ctypes.data,
        second_count.ctypes.data,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polygon_intersection_simplices")
    return PolygonIntersectionSimplices(
        simplices=simplices,
        simplex_counts=simplex_counts,
        area=area,
        first_moment=moment,
        status=status,
    )


def _tetrahedra(value: object, name: str, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (4, 3):
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n, 4, 3); got {_input_summary(value)}."
        )
    return array


def _capacity(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    return bounded_integer(value, name, lower=1, upper=2**31 - 1)


def tetrahedron_intersection_moments(
    first: object,
    second: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of tetrahedron pairs ``(n, 4, 3)``.

    Returns ``(volume (n,), first_moment (n, 3), status (n,))``. The intersection
    has at most eight faces, so the default vertex capacity (48) is never binding
    for valid input.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError(
            f"_meshcore.tetrahedron_intersection_moments.batches must have equal length; got {first_array.shape[0]} and {second_array.shape[0]}."
        )
    capacity = _capacity(vertex_capacity, 48, "vertex_capacity")
    count = first_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_moments"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        capacity,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_moments")
    return volume, moment, status


def tetrahedron_intersection_simplices(
    first: object,
    second: object,
    /,
    *,
    vertex_capacity: int | None = None,
    simplex_capacity: int | None = None,
) -> TetrahedronIntersectionSimplices:
    """Tetrahedron-pair intersections as tetrahedron partitions plus their moments.

    Returns :class:`TetrahedronIntersectionSimplices` with ``simplices (n, S, 4, 3), simplex_counts (n,), volume, first_moment,
    status``. Each nonempty intersection is coned from the vertex average of its
    vertices over the fan triangles of every outward face, the fan of
    ``tetrahedron_intersection_moments``, so the signed simplex volumes sum to
    ``volume`` term by term. The default capacity (20) bounds every intersection.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError(
            f"_meshcore.tetrahedron_intersection_simplices.batches must have equal length; got {first_array.shape[0]} and {second_array.shape[0]}."
        )
    vertices = _capacity(vertex_capacity, 48, "vertex_capacity")
    capacity = _capacity(simplex_capacity, 20, "simplex_capacity")
    count = first_array.shape[0]
    simplices = np.zeros((count, capacity, 4, 3), dtype=np.float64)
    simplex_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_tetrahedron_intersection_simplices"](
        count,
        first_array.ctypes.data,
        second_array.ctypes.data,
        vertices,
        capacity,
        simplices.ctypes.data,
        simplex_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "tetrahedron_intersection_simplices")
    return TetrahedronIntersectionSimplices(
        simplices=simplices,
        simplex_counts=simplex_counts,
        volume=volume,
        first_moment=moment,
        status=status,
    )


def _halfspaces(
    normals: object, offsets: object, plane_counts: object, width: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_array = _real_array(normals, "normals")
    if normal_array.ndim != 3 or normal_array.shape[2] != width:
        raise ValueError(
            f"_meshcore._halfspaces.normals must have shape (n, m, {width}); got {_input_summary(normals)}."
        )
    offset_array = _real_array(offsets, "offsets")
    if offset_array.shape != normal_array.shape[:2]:
        raise ValueError(
            f"_meshcore._halfspaces.offsets must have shape (n, m); got {_input_summary(offsets)}."
        )
    count_array = _index_array(plane_counts, "plane_counts")
    if count_array.shape != normal_array.shape[:1]:
        raise ValueError(
            f"_meshcore._halfspaces.plane_counts must have shape (n,); got {_input_summary(plane_counts)}."
        )
    if np.any(count_array < 0) or np.any(count_array > normal_array.shape[1]):
        raise ValueError(
            f"_meshcore._halfspaces.plane_counts must lie in [0, {normal_array.shape[1]}]; got {_input_summary(plane_counts)}."
        )
    return normal_array, offset_array, np.ascontiguousarray(count_array, dtype=np.int32)


def polyhedron_clip_moments(
    normals: object,
    offsets: object,
    plane_counts: object,
    tetrahedra: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moments of convex polyhedra ``{x : n . x <= h}`` intersected with tetrahedra.

    ``normals`` is ``(n, m, 3)``, ``offsets`` ``(n, m)``, ``plane_counts`` ``(n,)``
    and ``tetrahedra`` ``(n, 4, 3)``. Returns ``(volume, first_moment, status)``.
    The default vertex capacity ``4 (m + 4)`` bounds every intermediate polytope.
    """

    library = load_meshcore()
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    tetrahedron_array = _tetrahedra(tetrahedra, "tetrahedra")
    if tetrahedron_array.shape[0] != normal_array.shape[0]:
        raise ValueError(
            f"_meshcore.polyhedron_clip_moments.normals must have equal length; got {_input_summary(normals)}."
        )
    capacity = _capacity(
        vertex_capacity, 4 * (normal_array.shape[1] + 4), "vertex_capacity"
    )
    count = normal_array.shape[0]
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_polyhedron_clip_moments"](
        count,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        tetrahedron_array.ctypes.data,
        capacity,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "polyhedron_clip_moments")
    return volume, moment, status


def _box(lower: object, upper: object, width: int, /) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _real_array(lower, "box_lower")
    upper_array = _real_array(upper, "box_upper")
    if lower_array.shape != (width,) or upper_array.shape != (width,):
        raise ValueError(
            f"_meshcore._box.width must have shape ({width},); got {_input_summary(width)}."
        )
    if not np.all(lower_array < upper_array):
        raise ValueError(
            f"_meshcore._box.lower must be strictly below box_upper; got {_input_summary(lower)}."
        )
    return lower_array, upper_array


def clip_box_halfplanes(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> BoxHalfplaneClip:
    """Clip one axis-aligned 2D box by per-item halfplane sets ``{x : n . x <= h}``.

    Returns :class:`BoxHalfplaneClip` with ``vertices (n, V, 2), edge_labels (n, V), vertex_counts (n,), area (n,),
    first_moment (n, 2), status (n,)``. Edge ``k`` joins vertices ``k`` and
    ``k + 1`` (cyclic, counterclockwise) and is labeled with the item-local plane
    index or ``-(1 + 2 axis + side)`` for box sides. The default capacity
    ``m + 4`` is for convex polygons.
    """

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 2)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 2
    )
    capacity = _capacity(vertex_capacity, normal_array.shape[1] + 4, "vertex_capacity")
    count = normal_array.shape[0]
    vertices = np.zeros((count, capacity, 2), dtype=np.float64)
    labels = np.zeros((count, capacity), dtype=np.int32)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    area = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 2), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfplanes"](
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        capacity,
        vertices.ctypes.data,
        labels.ctypes.data,
        vertex_counts.ctypes.data,
        area.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfplanes")
    return BoxHalfplaneClip(
        vertices=vertices,
        edge_labels=labels,
        vertex_counts=vertex_counts,
        area=area,
        first_moment=moment,
        status=status,
    )


def clip_reference_triangle_exact(
    planes: tuple[tuple[Fraction, Fraction, Fraction], ...],
    /,
    *,
    maximum_scratch_bytes: int = 64 * 1024**2,
    maximum_work_units: int = 100_000,
) -> tuple[NDArray[np.int32], NDArray[np.int32], int]:
    """Native rational halfplanes; retain input boundary support pairs.

    Positive row denominator clearing and signed integer limb packing are .
    Native arbitrary-integer side tests decide every clipping transition. No
    rounded coefficient or vertex participates in topology. Source rows are
    (-1,0,0), (0,-1,0), (1,1,1), followed by three input target halfplanes.
    """
    if len(planes) != 6 or any(
        len(row) != 3 or any(not isinstance(value, Fraction) for value in row)
        for row in planes
    ):
        raise TypeError(
            f"_meshcore.clip_reference_triangle_exact.planes must satisfy the input contract (reference clipping requires six three-Fraction halfplanes.); got {_input_summary(planes)}."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 < value <= np.iinfo(np.int64).max
        for value in (maximum_scratch_bytes, maximum_work_units)
    ):
        raise ValueError(
            f"_meshcore.clip_reference_triangle_exact.planes must satisfy the input contract (reference clipping needs positive int64 scratch/work limits.); got {_input_summary(planes)}."
        )
    # The packed ABI payload and result storage allocate from the
    # native pool. Python Fraction/int temporaries are not measured by
    # that pool and remain the caller's separate host-coefficient budget.
    with NativeExecutionScope(
        NativeExecutionLimits(maximum_work_units, 0, 0, maximum_scratch_bytes, math.inf)
    ) as budget:
        allowance = budget.remaining()
        _raise_for_call(allowance.status, "clip_reference_triangle_exact_allowance")
        work_limit = min(maximum_work_units, allowance.remaining_work_units)
        budget.admit_work_bound(max(1, work_limit))
        integers: list[int] = []
        for row in planes:
            budget.charge(work=0)
            denominator = math.lcm(*(value.denominator for value in row))
            integers.extend(
                value.numerator * (denominator // value.denominator) for value in row
            )
        sizes = [(abs(value).bit_length() + 31) // 32 for value in integers]
        resident_bound = 16384 + max(1, max(sizes)) * 4096
        if resident_bound > maximum_scratch_bytes:
            raise MeshcoreError(
                MeshcoreStatus.CAPACITY_EXCEEDED,
                f"_meshcore.clip_reference_triangle_exact.planes must satisfy the input contract (reference integer clipping exceeds its scratch bound before packing.); got {_input_summary(planes)}.",
            )
        offsets = budget.allocate_host_array((19,), np.int64)
        signs = budget.allocate_host_array((18,), np.int8)
        packed = budget.allocate_host_array((sum(sizes),), np.uint32)
        offsets[0] = 0
        cursor = 0
        for index, value in enumerate(integers):
            budget.charge(work=0)
            signs[index] = int(value > 0) - int(value < 0)
            magnitude = abs(value)
            while magnitude:
                budget.charge(work=0)
                packed[cursor] = magnitude & 0xFFFFFFFF
                cursor += 1
                magnitude >>= 32
            offsets[index + 1] = cursor
        support = budget.allocate_host_array((6, 2), np.int32)
        labels = budget.allocate_host_array((6,), np.int32)
        count = budget.allocate_host_array((1,), np.int32)
        work = budget.allocate_host_array((1,), np.int64)
        count[0] = 0
        work[0] = 0
        allowance = budget.remaining()
        _raise_for_call(allowance.status, "clip_reference_triangle_exact_allowance")
        # The native ABI bounds each entire clipping phase before execution
        # and returns its measured arithmetic work even on partial refusal.
        # Admit that local cap first, then charge the consumed work exactly
        # once before publishing any result. Packing visits are clock-only,
        # not a new CPU-instruction proxy for the declared work quantity.
        scratch_limit = min(maximum_scratch_bytes, allowance.remaining_managed_bytes)
        if scratch_limit < resident_bound:
            raise MeshcoreError(
                MeshcoreStatus.CAPACITY_EXCEEDED,
                f"_meshcore.clip_reference_triangle_exact.planes must satisfy the input contract (reference integer clipping exceeds the remaining native scratch bound.); got {_input_summary(planes)}.",
            )
        library = load_meshcore()
        call = library["phx_mc_clip_reference_triangle_exact"](
            packed.ctypes.data,
            packed.size,
            offsets.ctypes.data,
            signs.ctypes.data,
            int(scratch_limit),
            int(work_limit),
            6,
            support.ctypes.data,
            labels.ctypes.data,
            count.ctypes.data,
            work.ctypes.data,
        )
        budget.charge(work=int(work[0]))
        _raise_for_call(call, "clip_reference_triangle_exact")
        return support[: int(count[0])], labels[: int(count[0])], int(work[0])


@final
class ExactReferenceTetrahedronClip(StrictModule):
    supporting_planes: NDArray[np.int32]
    incident_planes: NDArray[np.uint16]
    face_offsets: NDArray[np.int32]
    face_planes: NDArray[np.int32]
    face_vertices: NDArray[np.int32]
    work_units: int = eqx.field(static=True)


def clip_reference_tetrahedron_exact(
    planes: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...],
    /,
    *,
    maximum_scratch_bytes: int,
    maximum_work_units: int,
) -> ExactReferenceTetrahedronClip:
    """Clip the input reference tetrahedron with exact rational halfspaces.

    ``planes`` holds 4..16 rows of four Fractions ``(a, b, c, d)`` meaning
    a*x+b*y+c*z+d<=0 in original reference coordinates. The first four rows
    are positive multiples of (-1,0,0,0), (0,-1,0,0), (0,0,-1,0), (1,1,1,-1);
    the remaining rows are arbitrary exact halfspaces. With F rows the result
    has at most 2F-4 vertices, F faces and 6F-12 face entries: source support
    triplets, incident plane masks (bit p = plane row p) and oriented face CSR
    with face plane labels. Empty or lower-dimensional intersections return
    zero vertices and faces. Output is scientific plane incidence, never a
    rounded numerical replacement for input chart coefficients.
    """

    if (
        not 4 <= len(planes) <= 16
        or any(
            len(row) != 4 or any(not isinstance(value, Fraction) for value in row)
            for row in planes
        )
    ):
        raise TypeError(
            f"_meshcore.clip_reference_tetrahedron_exact.planes must satisfy the input contract (tetra clipping requires four to sixteen four-Fraction input planes.); got {_input_summary(planes)}."
        )
    plane_count = len(planes)
    vertex_capacity = 2 * plane_count - 4
    entry_capacity = 6 * plane_count - 12
    maximum_scratch_bytes = _limit(maximum_scratch_bytes, 1, "maximum_scratch_bytes")
    maximum_work_units = _limit(maximum_work_units, 1, "maximum_work_units")
    input_bits = max(
        sum(value.denominator.bit_length() for value in row)
        + max(value.numerator.bit_length() for value in row)
        for row in planes
    )
    # Covers up to 64 cleared integers of this many 30-bit digits plus their
    # lcm/product temporaries, so every admitted plane count fits.
    temporary_bytes = 64 * (256 + 8 * ((input_bits + 29) // 30))
    if temporary_bytes > maximum_scratch_bytes:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"_meshcore.clip_reference_tetrahedron_exact.planes must satisfy the input contract (Input tetra coefficients exceed host packing scratch before allocation.); got {_input_summary(planes)}.",
        )
    with (
        NativeExecutionScope(
            NativeExecutionLimits(
                maximum_work_units, 1, 0, maximum_scratch_bytes, math.inf
            )
        ) as budget,
        ExitStack() as preparation,
    ):
        host = current_native_host_workspace()
        if host is None:
            host = preparation.enter_context(budget.host_workspace())
            host.retain_owner(planes)
        starting = host.bound
        host.set_bound(starting + temporary_bytes)
        try:
            integers: list[int] = []
            for row in planes:
                budget.charge(work=0)
                denominator = math.lcm(*(value.denominator for value in row))
                integers.extend(
                    value.numerator * (denominator // value.denominator) for value in row
                )
            sizes = [(abs(value).bit_length() + 31) // 32 for value in integers]
            # Native pre-allocation resident bound: 2048 + 4096*M bytes per
            # bounded vertex slot (unchanged 32768 + 65536*M for ten planes).
            resident = vertex_capacity * (2048 + 4096 * max(1, max(sizes)))
            if resident > maximum_scratch_bytes:
                raise MeshcoreError(
                    MeshcoreStatus.CAPACITY_EXCEEDED,
                    f"_meshcore.clip_reference_tetrahedron_exact.planes must satisfy the input contract (Input tetra predicates exceed native scratch before packing.); got {_input_summary(planes)}.",
                )
            offsets = budget.allocate_host_array((4 * plane_count + 1,), np.int64)
            signs = budget.allocate_host_array((4 * plane_count,), np.int8)
            packed = budget.allocate_host_array((sum(sizes),), np.uint32)
            offsets[0] = 0
            cursor = 0
            for row, value in enumerate(integers):
                signs[row] = int(value > 0) - int(value < 0)
                magnitude = abs(value)
                while magnitude:
                    budget.charge(work=0)
                    packed[cursor] = magnitude & 0xFFFFFFFF
                    cursor += 1
                    magnitude >>= 32
                offsets[row + 1] = cursor
            supports = budget.allocate_host_array((vertex_capacity, 3), np.int32)
            incidence = budget.allocate_host_array((vertex_capacity,), np.uint16)
            face_offsets = budget.allocate_host_array((plane_count + 1,), np.int32)
            face_offsets[0] = 0
            face_planes = budget.allocate_host_array((plane_count,), np.int32)
            face_vertices = budget.allocate_host_array((entry_capacity,), np.int32)
            counts = budget.allocate_host_array((2,), np.int32)
            work = budget.allocate_host_array((1,), np.int64)
            counts.fill(0)
            work.fill(0)
            allowance = budget.remaining()
            _raise_for_call(
                allowance.status, "clip_reference_tetrahedron_exact_allowance"
            )
            scratch = min(maximum_scratch_bytes, allowance.remaining_managed_bytes)
            if scratch < resident:
                raise MeshcoreError(
                    MeshcoreStatus.CAPACITY_EXCEEDED,
                    f"_meshcore.clip_reference_tetrahedron_exact.planes must satisfy the input contract (Input tetra predicates exceed remaining native scratch.); got {_input_summary(planes)}.",
                )
            work_limit = min(maximum_work_units, allowance.remaining_work_units)
            budget.admit_work_bound(max(1, work_limit))
            library = load_meshcore()
            call = library["phx_mc_clip_reference_tetrahedron_exact"](
                ctypes.byref(
                    _ClipReferenceTetrahedronExactRequest(
                        words=packed.ctypes.data,
                        word_count=packed.size,
                        offsets=offsets.ctypes.data,
                        signs=signs.ctypes.data,
                        scratch_limit=scratch,
                        work_limit=work_limit,
                        plane_count=plane_count,
                        vertex_capacity=vertex_capacity,
                        face_capacity=plane_count,
                        entry_capacity=entry_capacity,
                        supporting_planes=supports.ctypes.data,
                        incident_planes=incidence.ctypes.data,
                        face_offsets=face_offsets.ctypes.data,
                        face_planes=face_planes.ctypes.data,
                        face_vertices=face_vertices.ctypes.data,
                        vertex_count=counts.ctypes.data,
                        face_count=counts[1:].ctypes.data,
                        work_units=work.ctypes.data,
                    )
                )
            )
            # Native decoding/predicates charge the same scope themselves.
            _raise_for_call(call, "clip_reference_tetrahedron_exact")
            nv, nf = int(counts[0]), int(counts[1])
            entries = int(face_offsets[nf]) if nf else 0
            return ExactReferenceTetrahedronClip(
                supports[:nv],
                incidence[:nv],
                face_offsets[: nf + 1],
                face_planes[:nf],
                face_vertices[:entries],
                int(work[0]),
            )
        finally:
            host.set_bound(starting)


@final
class ExactTetrahedronArrangement(StrictModule):
    """Exact pooled arrangement of positive tetrahedra under a clipping DAG.

    All geometry is in the single input frame. ``vertex_words``,
    ``vertex_word_offsets`` (4V+1) and ``vertex_signs`` (4V) pack each
    vertex's exact homogeneous integers (X, Y, Z, W), W > 0, as signed
    little-endian uint32 magnitudes. Vertices are welded by exact projective
    equality and numbered by first use; ``vertex_input_index`` is the
    smallest input point proven equal to a vertex, else -1. Job j owns faces
    ``cell_face_offsets[j]:cell_face_offsets[j+1]``; face f's outward,
    strictly convex loop ``face_vertices[face_offsets[f]:face_offsets[f+1]]``
    starts at its smallest vertex and ``face_planes[f]`` is the signed 1-based
    plane reference whose <= 0 side holds the cell. Empty and zero-measure
    jobs have no faces. ``reuse_counters`` (propagated empty jobs, sections,
    side evaluations, side reuses, vertex constructions, construction reuses,
    vertex welds, pool vertices) and ``evidence`` (work, primitive queries,
    peak managed bytes, completed jobs, failure category, failure row) are
    native evidence; ``work_units`` is the call's total native work. Numeric
    vertex and plane ids carry no scientific identity.
    """

    vertex_words: NDArray[np.uint32]
    vertex_word_offsets: NDArray[np.int64]
    vertex_signs: NDArray[np.int8]
    vertex_input_index: NDArray[np.int32]
    cell_face_offsets: NDArray[np.int64]
    face_offsets: NDArray[np.int64]
    face_vertices: NDArray[np.int32]
    face_planes: NDArray[np.int32]
    reuse_counters: NDArray[np.int64]
    evidence: NDArray[np.int64]
    work_units: int = eqx.field(static=True)


# Fixed record widths of phx_mc_arrange_tetrahedra_exact (phydrax_meshcore.h).
_TETRAHEDRA_ARRANGEMENT_EVIDENCE = 6
_TETRAHEDRA_ARRANGEMENT_SIZES = 5
_TETRAHEDRA_ARRANGEMENT_COUNTERS = 8


def _exact_arrangement_rows(rows: object, width: int, name: str, /) -> None:
    if not isinstance(rows, tuple) or any(
        not isinstance(row, tuple)
        or len(row) != width
        or any(not isinstance(value, Fraction) for value in row)
        for row in rows
    ):
        raise TypeError(
            f"_meshcore.arrange_tetrahedra_exact.{name} must be a tuple of {width}-Fraction tuples; got {_input_summary(rows)}."
        )


def _exact_arrangement_indices(
    value: object, name: str, shape: tuple[int | None, ...], dtype: type[np.integer], /
) -> np.ndarray:
    array = _index_array(value, f"arrange_tetrahedra_exact.{name}")
    limits = np.iinfo(dtype)
    if (
        array.ndim != len(shape)
        or any(
            size is not None and actual != size
            for actual, size in zip(array.shape, shape, strict=True)
        )
        or (array.size and (array.min() < limits.min or array.max() > limits.max))
    ):
        raise ValueError(
            f"_meshcore.arrange_tetrahedra_exact.{name} must have shape {shape} and fit {np.dtype(dtype).name}; got {_input_summary(value)}."
        )
    return array


def arrange_tetrahedra_exact(
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    planes: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...],
    tetrahedra: ArrayLike,
    tetrahedron_planes: ArrayLike,
    parents: ArrayLike,
    cut_offsets: ArrayLike,
    cut_planes: ArrayLike,
    /,
    *,
    maximum_scratch_bytes: int,
    maximum_work_units: int,
) -> ExactTetrahedronArrangement:
    """Arrange positive tetrahedra under a deterministic clipping DAG, exactly.

    ``points`` are exact rational points and ``planes`` canonical exact rows
    ``(a, b, c, d)`` of ``a*x+b*y+c*z+d = 0`` (nonzero normal, first nonzero
    coefficient positive), both in ONE frame; each row is packed once to
    integers (points over their positive lcm denominator, planes cleared and
    divided by their gcd, which preserves orientation). Plane references are
    signed 1-based: ``r > 0`` selects ``row <= 0`` and ``r < 0`` its
    negation. ``tetrahedra`` (S, 4) are strictly positive point rows and
    ``tetrahedron_planes`` (S, 4) slot k their outward face opposite corner
    k. Job j starts at tetrahedron s for ``parents[j] = -(s+1)`` or at the
    result of an earlier job ``parents[j] < j`` and intersects the halfspaces
    ``cut_planes[cut_offsets[j]:cut_offsets[j+1]]``; each inherited chain has
    at most 16 halfspaces. Children of empty jobs stay empty without
    clipping. The native owner validates everything before publication,
    charges one primitive query per job and pools exact plane sides,
    crossings and welded points across jobs; a refusal publishes no result
    and the nested execution scope keeps its partial work evidence.
    """

    _exact_arrangement_rows(points, 3, "points")
    _exact_arrangement_rows(planes, 4, "planes")
    tetrahedron_rows = _exact_arrangement_indices(
        tetrahedra, "tetrahedra", (None, 4), np.int32
    )
    tetrahedron_count = tetrahedron_rows.shape[0]
    face_rows = _exact_arrangement_indices(
        tetrahedron_planes, "tetrahedron_planes", (tetrahedron_count, 4), np.int32
    )
    parent_rows = _exact_arrangement_indices(parents, "parents", (None,), np.int32)
    job_count = parent_rows.shape[0]
    offset_rows = _exact_arrangement_indices(
        cut_offsets, "cut_offsets", (job_count + 1,), np.int64
    )
    cut_rows = _exact_arrangement_indices(cut_planes, "cut_planes", (None,), np.int32)
    maximum_scratch_bytes = _limit(maximum_scratch_bytes, 1, "maximum_scratch_bytes")
    maximum_work_units = _limit(maximum_work_units, 1, "maximum_work_units")
    # Python integers, limb chunks and their joined buffer for every packed
    # scalar: at most 320 + 16 bytes per 30-bit digit of the row bound.
    temporary_bytes = 4096
    for row in (*points, *planes):
        row_bits = sum(value.denominator.bit_length() for value in row) + max(
            value.numerator.bit_length() for value in row
        )
        temporary_bytes += (len(row) + 1) * (320 + 16 * ((row_bits + 29) // 30))
    if temporary_bytes > maximum_scratch_bytes:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"_meshcore.arrange_tetrahedra_exact.points must satisfy the input contract (exact rows exceed host packing scratch before allocation.); got {_input_summary(points)}.",
        )
    with (
        NativeExecutionScope(
            NativeExecutionLimits(
                maximum_work_units, job_count, 0, maximum_scratch_bytes, math.inf
            )
        ) as budget,
        ExitStack() as preparation,
    ):
        host = current_native_host_workspace()
        if host is None:
            host = preparation.enter_context(budget.host_workspace())
            host.retain_owner(
                (
                    points,
                    planes,
                    tetrahedra,
                    tetrahedron_planes,
                    parents,
                    cut_offsets,
                    cut_planes,
                )
            )
        starting = host.bound
        host.set_bound(starting + temporary_bytes)
        try:

            def pack(
                integers: list[int],
            ) -> tuple[NDArray[np.uint32], NDArray[np.int64], NDArray[np.int8]]:
                offsets = budget.allocate_host_array((len(integers) + 1,), np.int64)
                signs = budget.allocate_host_array((len(integers),), np.int8)
                chunks: list[bytes] = []
                cursor = 0
                offsets[0] = 0
                for scalar, value in enumerate(integers):
                    if scalar % 1024 == 0:
                        budget.charge(work=0)
                    magnitude = abs(value)
                    limbs = (magnitude.bit_length() + 31) // 32
                    signs[scalar] = int(value > 0) - int(value < 0)
                    if limbs:
                        chunks.append(magnitude.to_bytes(4 * limbs, "little"))
                    cursor += limbs
                    offsets[scalar + 1] = cursor
                words = budget.allocate_host_array((cursor,), np.uint32)
                if cursor:
                    words[:] = np.frombuffer(b"".join(chunks), dtype="<u4")
                return words, offsets, signs

            point_integers: list[int] = []
            for index, (x, y, z) in enumerate(points):
                if index % 1024 == 0:
                    budget.charge(work=0)
                scale = math.lcm(x.denominator, y.denominator, z.denominator)
                point_integers.extend(
                    (
                        x.numerator * (scale // x.denominator),
                        y.numerator * (scale // y.denominator),
                        z.numerator * (scale // z.denominator),
                        scale,
                    )
                )
            plane_integers: list[int] = []
            for index, row in enumerate(planes):
                if index % 1024 == 0:
                    budget.charge(work=0)
                scale = math.lcm(*(value.denominator for value in row))
                cleared = [
                    value.numerator * (scale // value.denominator) for value in row
                ]
                divisor = math.gcd(*cleared) or 1
                plane_integers.extend(value // divisor for value in cleared)
            point_words, point_offsets, point_signs = pack(point_integers)
            plane_words, plane_offsets, plane_signs = pack(plane_integers)
            del point_integers, plane_integers
            tetrahedron_buffer = budget.allocate_host_array(
                (tetrahedron_count, 4), np.int32
            )
            tetrahedron_buffer[...] = tetrahedron_rows
            face_buffer = budget.allocate_host_array((tetrahedron_count, 4), np.int32)
            face_buffer[...] = face_rows
            parent_buffer = budget.allocate_host_array((job_count,), np.int32)
            parent_buffer[...] = parent_rows
            offset_buffer = budget.allocate_host_array((job_count + 1,), np.int64)
            offset_buffer[...] = offset_rows
            cut_buffer = budget.allocate_host_array((cut_rows.shape[0],), np.int32)
            cut_buffer[...] = cut_rows
            evidence = budget.allocate_host_array(
                (_TETRAHEDRA_ARRANGEMENT_EVIDENCE,), np.int64
            )
            evidence.fill(0)
            allowance = budget.remaining()
            _raise_for_call(allowance.status, "arrange_tetrahedra_exact_allowance")
            scratch = min(maximum_scratch_bytes, allowance.remaining_managed_bytes)
            work_limit = min(maximum_work_units, allowance.remaining_work_units)
            if scratch <= 0 or work_limit <= 0:
                raise MeshcoreError(
                    MeshcoreStatus.CAPACITY_EXCEEDED,
                    f"_meshcore.arrange_tetrahedra_exact.points must satisfy the input contract (no native scratch or work remains before arrangement.); got {_input_summary(points)}.",
                )
            budget.admit_work_bound(work_limit)
            library = load_meshcore()
            handle = ctypes.c_void_p()
            call = library["phx_mc_arrange_tetrahedra_exact"](
                ctypes.byref(
                    _ArrangeTetrahedraExactRequest(
                        point_count=len(points),
                        plane_count=len(planes),
                        tetrahedron_count=tetrahedron_count,
                        job_count=job_count,
                        cut_count=cut_rows.shape[0],
                        point_word_count=point_words.size,
                        plane_word_count=plane_words.size,
                        point_words=point_words.ctypes.data,
                        point_offsets=point_offsets.ctypes.data,
                        point_signs=point_signs.ctypes.data,
                        plane_words=plane_words.ctypes.data,
                        plane_offsets=plane_offsets.ctypes.data,
                        plane_signs=plane_signs.ctypes.data,
                        tetrahedra=tetrahedron_buffer.ctypes.data,
                        tetrahedron_planes=face_buffer.ctypes.data,
                        parents=parent_buffer.ctypes.data,
                        cut_offsets=offset_buffer.ctypes.data,
                        cut_planes=cut_buffer.ctypes.data,
                        scratch_limit=scratch,
                        work_limit=work_limit,
                        evidence=evidence.ctypes.data,
                        result=ctypes.addressof(handle),
                    )
                )
            )
            try:
                # Native decoding/predicates charge this scope themselves; a
                # refusal leaves its work in the scope evidence.
                _raise_for_call(
                    call,
                    "arrange_tetrahedra_exact"
                    f"[failure category {int(evidence[4])} row {int(evidence[5])}"
                    f" after {int(evidence[3])} jobs, work {int(evidence[0])}"
                    f"/{work_limit}, native scratch peak {int(evidence[2])}"
                    f"/{scratch} bytes]",
                )
                if not handle.value:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        f"_meshcore.arrange_tetrahedra_exact.points must satisfy the input contract (missing arrangement result); got {_input_summary(points)}.",
                    )
                sizes = budget.allocate_host_array(
                    (_TETRAHEDRA_ARRANGEMENT_SIZES,), np.int64
                )
                _raise_for_call(
                    library["phx_mc_tetrahedra_arrangement_sizes"](
                        handle, sizes.ctypes.data
                    ),
                    "tetrahedra_arrangement_sizes",
                )
                vertices, words, jobs, faces, entries = (int(size) for size in sizes)
                if jobs != job_count or min(vertices, words, faces, entries) < 0:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        f"_meshcore.arrange_tetrahedra_exact.points must satisfy the input contract (arrangement sizes disagree with the request); got {_input_summary(points)}.",
                    )
                result = ExactTetrahedronArrangement(
                    budget.allocate_host_array((words,), np.uint32),
                    budget.allocate_host_array((4 * vertices + 1,), np.int64),
                    budget.allocate_host_array((4 * vertices,), np.int8),
                    budget.allocate_host_array((vertices,), np.int32),
                    budget.allocate_host_array((job_count + 1,), np.int64),
                    budget.allocate_host_array((faces + 1,), np.int64),
                    budget.allocate_host_array((entries,), np.int32),
                    budget.allocate_host_array((faces,), np.int32),
                    budget.allocate_host_array(
                        (_TETRAHEDRA_ARRANGEMENT_COUNTERS,), np.int64
                    ),
                    evidence,
                    int(evidence[0]),
                )
                _raise_for_call(
                    library["phx_mc_tetrahedra_arrangement_export"](
                        handle,
                        result.vertex_words.ctypes.data,
                        result.vertex_word_offsets.ctypes.data,
                        result.vertex_signs.ctypes.data,
                        result.vertex_input_index.ctypes.data,
                        result.cell_face_offsets.ctypes.data,
                        result.face_offsets.ctypes.data,
                        result.face_vertices.ctypes.data,
                        result.face_planes.ctypes.data,
                        result.reuse_counters.ctypes.data,
                    ),
                    "tetrahedra_arrangement_export",
                )
            finally:
                if handle.value:
                    library["phx_mc_tetrahedra_arrangement_free"](handle)
            for array in (
                result.vertex_words,
                result.vertex_word_offsets,
                result.vertex_signs,
                result.vertex_input_index,
                result.cell_face_offsets,
                result.face_offsets,
                result.face_vertices,
                result.face_planes,
                result.reuse_counters,
                result.evidence,
            ):
                array.flags.writeable = False
            return result
        finally:
            host.set_bound(starting)


def polyhedron_corner_hexes(
    edges: NDArray[np.int64],
    face_offsets: NDArray[np.int64],
    face_vertices: NDArray[np.int64],
    cell_offsets: NDArray[np.int64],
    cell_faces: NDArray[np.int64],
    vertex_offsets: NDArray[np.int64],
    cell_vertices: NDArray[np.int64],
    corner_orientations: NDArray[np.int8],
    original_vertex_count: int,
    /,
    *,
    maximum_cells: int,
    maximum_scratch_bytes: int,
    maximum_work_units: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Construct corner-link templates from source-cell incidence."""
    for name, value in (
        ("edges", edges),
        ("face_offsets", face_offsets),
        ("face_vertices", face_vertices),
        ("cell_offsets", cell_offsets),
        ("cell_faces", cell_faces),
        ("vertex_offsets", vertex_offsets),
        ("cell_vertices", cell_vertices),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.dtype(np.int64)
            or not value.flags.c_contiguous
        ):
            raise TypeError(
                f"{name} must be a contiguous int64 incidence array; got {type(value).__qualname__}."
            )
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or any(
            value.ndim != 1
            for value in (
                face_offsets,
                face_vertices,
                cell_offsets,
                cell_faces,
                vertex_offsets,
                cell_vertices,
            )
        )
    ):
        raise ValueError(
            f"_meshcore.polyhedron_corner_hexes.edges must satisfy the input contract (Cut corner incidence requires edge pairs and one-dimensional CSR banks.); got {_input_summary(edges)}."
        )
    if (
        not isinstance(corner_orientations, np.ndarray)
        or corner_orientations.dtype != np.dtype(np.int8)
        or not corner_orientations.flags.c_contiguous
        or corner_orientations.shape != cell_vertices.shape
    ):
        raise TypeError(
            f"_meshcore.polyhedron_corner_hexes.edges must bind every input cell-vertex occurrence; got {_input_summary(edges)}."
        )
    nv = _limit(original_vertex_count, 1, "original_vertex_count")
    maximum_cells = _limit(maximum_cells, 1, "maximum_cells")
    maximum_scratch_bytes = _limit(maximum_scratch_bytes, 1, "maximum_scratch_bytes")
    maximum_work_units = _limit(maximum_work_units, 1, "maximum_work_units")
    nf, nc = face_offsets.size - 1, cell_offsets.size - 1
    if nf <= 0 or nc <= 0 or vertex_offsets.size != nc + 1:
        raise ValueError(
            f"_meshcore.polyhedron_corner_hexes.edges must retain entire input source faces and cells; got {_input_summary(edges)}."
        )
    count = cell_vertices.size
    if count > maximum_cells:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"polyhedron_corner_hexes.count must not exceed {maximum_cells}; got {count}.",
        )
    with NativeExecutionScope(
        NativeExecutionLimits(maximum_work_units, 0, 0, maximum_scratch_bytes, math.inf)
    ) as budget:
        output = budget.allocate_host_array((count, 8), np.int64)
        parents = budget.allocate_host_array((count,), np.int64)
        completed = budget.allocate_host_array((1,), np.int64)
        witness = budget.allocate_host_array((4,), np.int64)
        work = budget.allocate_host_array((1,), np.int64)
        completed.fill(0)
        work.fill(0)
        allowance = budget.remaining()
        _raise_for_call(allowance.status, "polyhedron_corner_hexes_allowance")
        budget.admit_work_bound(
            max(1, min(maximum_work_units, allowance.remaining_work_units))
        )
        library = load_meshcore()
        call = library["phx_mc_polyhedron_corner_hexes"](
            ctypes.byref(
                _PolyhedronCornerHexesRequest(
                    nv=nv,
                    ne=edges.shape[0],
                    nf=nf,
                    nc=nc,
                    face_entries=face_vertices.size,
                    cell_face_entries=cell_faces.size,
                    corner_entries=cell_vertices.size,
                    edges=edges.ctypes.data,
                    face_offsets=face_offsets.ctypes.data,
                    face_vertices=face_vertices.ctypes.data,
                    cell_offsets=cell_offsets.ctypes.data,
                    cell_faces=cell_faces.ctypes.data,
                    vertex_offsets=vertex_offsets.ctypes.data,
                    cell_vertices=cell_vertices.ctypes.data,
                    corner_orientations=corner_orientations.ctypes.data,
                    vertex_count=nv + edges.shape[0] + nf + nc,
                    hex_capacity=maximum_cells,
                    scratch_limit=min(
                        maximum_scratch_bytes, allowance.remaining_managed_bytes
                    ),
                    work_limit=min(maximum_work_units, allowance.remaining_work_units),
                    hexes=output.ctypes.data,
                    parent_cells=parents.ctypes.data,
                    hex_count=completed.ctypes.data,
                    witness=witness.ctypes.data,
                    work_units=work.ctypes.data,
                )
            )
        )
        if call == MeshcoreStatus.REFINEMENT_LIMIT:
            raise MeshcoreError(
                MeshcoreStatus.REFINEMENT_LIMIT,
                f"_meshcore.polyhedron_corner_hexes.edges must satisfy the input contract (Input cut corner link (cell row, vertex row, edge degree, face degree) {tuple((int(value) for value in witness))} requires conforming grid-link regularization.); got {_input_summary(edges)}.",
            )
        _raise_for_call(call, "polyhedron_corner_hexes")
        return output[: int(completed[0])], parents[: int(completed[0])], int(work[0])


def clip_box_halfspaces(
    box_lower: object,
    box_upper: object,
    normals: object,
    offsets: object,
    plane_counts: object,
    /,
    *,
    vertex_capacity: int | None = None,
    face_capacity: int | None = None,
    face_vertex_capacity: int | None = None,
) -> BoxHalfspaceClip:
    """Clip one axis-aligned 3D box by per-item halfspace sets ``{x : n . x <= h}``.

    Returns :class:`BoxHalfspaceClip` with ``vertices (n, V, 3), vertex_counts (n,), face_offsets (n, F + 1),
    face_labels (n, F), face_vertices (n, W), face_counts (n,), volume (n,),
    first_moment (n, 3), status (n,)``. Faces are sorted by label, oriented
    counterclockwise from outside, and start at their smallest vertex. Default
    capacities follow the convex-polytope bounds ``F <= m + 6``, ``V <= 2F - 4``
    and ``W <= 2 (V + F - 2)`` (twice the edge count).
    """

    library = load_meshcore()
    lower, upper = _box(box_lower, box_upper, 3)
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    faces_bound = normal_array.shape[1] + 6
    vertices_bound = 2 * faces_bound - 4
    face_capacity_ = _capacity(face_capacity, faces_bound, "face_capacity")
    vertex_capacity_ = _capacity(vertex_capacity, vertices_bound, "vertex_capacity")
    face_vertex_capacity_ = _capacity(
        face_vertex_capacity,
        2 * (vertices_bound + faces_bound - 2),
        "face_vertex_capacity",
    )
    count = normal_array.shape[0]
    vertices = np.zeros((count, vertex_capacity_, 3), dtype=np.float64)
    vertex_counts = np.zeros((count,), dtype=np.int32)
    face_offsets = np.zeros((count, face_capacity_ + 1), dtype=np.int32)
    face_labels = np.zeros((count, face_capacity_), dtype=np.int32)
    face_vertices = np.zeros((count, face_vertex_capacity_), dtype=np.int32)
    face_counts = np.zeros((count,), dtype=np.int32)
    volume = np.zeros((count,), dtype=np.float64)
    moment = np.zeros((count, 3), dtype=np.float64)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_clip_box_halfspaces"](
        ctypes.byref(
            _ClipBoxHalfspacesRequest(
                count=count,
                box_lower=lower.ctypes.data,
                box_upper=upper.ctypes.data,
                plane_capacity=normal_array.shape[1],
                normals=normal_array.ctypes.data,
                offsets=offset_array.ctypes.data,
                plane_counts=count_array.ctypes.data,
                vertex_capacity=vertex_capacity_,
                face_capacity=face_capacity_,
                face_vertex_capacity=face_vertex_capacity_,
                vertices=vertices.ctypes.data,
                vertex_counts=vertex_counts.ctypes.data,
                face_offsets=face_offsets.ctypes.data,
                face_labels=face_labels.ctypes.data,
                face_vertices=face_vertices.ctypes.data,
                face_counts=face_counts.ctypes.data,
                volumes=volume.ctypes.data,
                first_moments=moment.ctypes.data,
                item_status=status.ctypes.data,
            )
        )
    )
    _raise_for_call(call, "clip_box_halfspaces")
    return BoxHalfspaceClip(
        vertices=vertices,
        vertex_counts=vertex_counts,
        face_offsets=face_offsets,
        face_labels=face_labels,
        face_vertices=face_vertices,
        face_counts=face_counts,
        volume=volume,
        first_moment=moment,
        status=status,
    )


# ---------------------------------------------------------------- contacts in 3D


def _simplices(value: object, name: str, corners: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (corners, 3):
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n, {corners}, 3); got {_input_summary(value)}."
        )
    return array


def _vertex_ids(
    first_ids: object | None,
    second_ids: object | None,
    shapes: tuple[tuple[int, int], tuple[int, int]],
    /,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if first_ids is None and second_ids is None:
        return None, None
    if first_ids is None or second_ids is None:
        raise ValueError(
            f"_meshcore._vertex_ids.first_ids must be given for both inputs or for neither; got {_input_summary(first_ids)}."
        )
    first = _index_array(first_ids, "first_ids")
    second = _index_array(second_ids, "second_ids")
    if first.shape != shapes[0] or second.shape != shapes[1]:
        raise ValueError(
            f"_meshcore._vertex_ids.first_ids must have shapes {shapes[0]} and {shapes[1]}; got {_input_summary(first_ids)}."
        )
    return (
        np.ascontiguousarray(first, dtype=np.int64),
        np.ascontiguousarray(second, dtype=np.int64),
    )


def _pointer(array: np.ndarray | None, /) -> int | None:
    return None if array is None else array.ctypes.data


def _contacts(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    ids: tuple[np.ndarray | None, np.ndarray | None],
    capacity: int | None,
    /,
) -> tuple[np.ndarray, ...]:
    library = load_meshcore()
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"_meshcore._contacts.name must have equal length; got {_input_summary(name)}."
        )
    count = first.shape[0]
    classes = np.zeros((count,), dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    if capacity is None:
        construction: tuple[np.ndarray, ...] = ()
        pointers: tuple[int | None, ...] = (None, None, None, None)
    else:
        construction = (
            np.zeros((count,), dtype=np.int32),
            np.zeros((count, capacity, 3), dtype=np.float64),
            np.zeros((count, capacity), dtype=np.float64),
            np.zeros((count, capacity, 2), dtype=np.int8),
        )
        pointers = tuple(array.ctypes.data for array in construction)
    call = library[name](
        count,
        first.ctypes.data,
        second.ctypes.data,
        _pointer(ids[0]),
        _pointer(ids[1]),
        classes.ctypes.data,
        *pointers,
        status.ctypes.data,
    )
    _raise_for_call(call, name.removeprefix("phx_mc_"))
    return (classes, *construction, status)


def triangle_intersection_classes(
    first: object,
    second: object,
    /,
    *,
    first_ids: object | None = None,
    second_ids: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:class:`IntersectionClass` of triangle pairs ``(n, 3, 3)``.

    Returns ``(classes (n,) int8, status (n,) int32)``. Optional vertex ids
    ``(n, 3)`` for both inputs make adjacency identity-based: a vertex contact
    is shared only when the ids agree, so coincident positions with distinct
    ids are ``TOUCHING``; ids repeated within a triangle or equal across
    triangles at distinct positions report ``INVALID_INPUT``. Without ids,
    coincident positions are shared. Failed items (``NONFINITE_INPUT``,
    ``RANGE_ERROR``, ``DEGENERATE_INPUT`` for collinear triangles,
    ``INVALID_INPUT``) have class -1.
    """

    first_array = _simplices(first, "first", 3)
    second_array = _simplices(second, "second", 3)
    ids = _vertex_ids(
        first_ids, second_ids, ((first_array.shape[0], 3), (second_array.shape[0], 3))
    )
    classes, status = _contacts(
        "phx_mc_triangle_intersections", first_array, second_array, ids, None
    )
    return classes, status


def triangle_intersections(
    first: object,
    second: object,
    /,
    *,
    first_ids: object | None = None,
    second_ids: object | None = None,
) -> ContactIntersections:
    """Contact classes and bounded contact constructions of triangle pairs.

    Returns :class:`ContactIntersections` with ``classes (n,), point_counts (n,), points (n, 6, 3), point_bounds
    (n, 6), point_features (n, 6, 2), status (n,))`` with the classes, ids and
    statuses of :func:`triangle_intersection_classes`. The intersection is
    empty, a point, a segment (endpoints ordered along the intersection line)
    or a convex polygon of at most six vertices, counterclockwise about the
    first triangle's normal. ``point_features[i, k]`` holds the feature of the
    first and second triangle containing point ``k`` exactly: 0-2 vertex ``j``,
    3-5 the edge opposite vertex ``j - 3``, 6 the relative interior. Points
    that are input vertices are (bound 0); the others are constructions
    ``p + t (q - p)`` with ``t`` a ratio of orientation expansions and
    ``point_bounds`` a rigorous max-norm distance bound from the point.
    Unused slots hold zero coordinates and features -1.
    """

    first_array = _simplices(first, "first", 3)
    second_array = _simplices(second, "second", 3)
    ids = _vertex_ids(
        first_ids, second_ids, ((first_array.shape[0], 3), (second_array.shape[0], 3))
    )
    classes, counts, points, bounds, features, status = _contacts(
        "phx_mc_triangle_intersections", first_array, second_array, ids, 6
    )
    return ContactIntersections(
        classes=classes,
        point_counts=counts,
        points=points,
        point_bounds=bounds,
        point_features=features,
        status=status,
    )


def segment_triangle_intersections(
    segments: object,
    triangles: object,
    /,
    *,
    segment_ids: object | None = None,
    triangle_ids: object | None = None,
) -> ContactIntersections:
    """Contact classes and constructions of segments ``(n, 2, 3)`` with triangles.

    Returns :class:`ContactIntersections` with ``classes (n,), point_counts (n,), points (n, 2, 3), point_bounds
    (n, 2), point_features (n, 2, 2), status (n,))`` as
    :func:`triangle_intersections`; segment features are 0 and 1 for the
    endpoints and 6 for the interior. ``SHARED_EDGE`` means the segment is an
    edge of the triangle, ``COPLANAR_OVERLAP`` a coplanar contact through the
    triangle's interior. A zero-length segment or collinear triangle reports
    ``DEGENERATE_INPUT``.
    """

    segment_array = _simplices(segments, "segments", 2)
    triangle_array = _simplices(triangles, "triangles", 3)
    ids = _vertex_ids(
        segment_ids,
        triangle_ids,
        ((segment_array.shape[0], 2), (triangle_array.shape[0], 3)),
    )
    classes, counts, points, bounds, features, status = _contacts(
        "phx_mc_segment_triangle_intersections", segment_array, triangle_array, ids, 2
    )
    return ContactIntersections(
        classes=classes,
        point_counts=counts,
        points=points,
        point_bounds=bounds,
        point_features=features,
        status=status,
    )


def point_triangle_locations(
    points: object, triangles: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Location of points ``(n, 3)`` relative to triangles ``(n, 3, 3)``.

    Returns ``(sides (n,) int8, features (n,) int8, status (n,) int32)``:
    ``sides`` is ``orient3d(t0, t1, t2, x)`` and ``features`` the triangle
    feature containing the point (0-2 vertex, 3-5 edge opposite vertex ``j - 3``,
    6 interior) or -1 when it is off the closed triangle.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 3)
    triangle_array = _simplices(triangles, "triangles", 3)
    if point_array.shape[0] != triangle_array.shape[0]:
        raise ValueError(
            f"_meshcore.point_triangle_locations.points must have equal length; got {_input_summary(points)}."
        )
    count = point_array.shape[0]
    sides = np.zeros((count,), dtype=np.int8)
    features = np.zeros((count,), dtype=np.int8)
    status = np.zeros((count,), dtype=np.int32)
    call = library["phx_mc_point_triangle_locations"](
        count,
        point_array.ctypes.data,
        triangle_array.ctypes.data,
        sides.ctypes.data,
        features.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "point_triangle_locations")
    return sides, features, status


# ---------------------------------------------------------------- triangulations


def _read_mesh(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    /,
    *,
    admit_host_output: Callable[[int], None] | None = None,
) -> NativeMeshArrays:
    # The native handle is released even when a copy fails.
    try:
        dimension = library["phx_mc_mesh_dimension"](handle)
        point_count = library["phx_mc_mesh_point_count"](handle)
        cell_count = library["phx_mc_mesh_cell_count"](handle)
        input_count = library["phx_mc_mesh_input_point_count"](handle)
        if admit_host_output is not None:
            # Five independent NumPy exports and their owner; native mesh
            # payload remains separately charged until the finally below.
            admit_host_output(
                6 * 256
                + point_count * dimension * 8
                + 2 * cell_count * (dimension + 1) * 4
                + input_count * 4
                + cell_count * 4
            )
        points = np.zeros((point_count, dimension), dtype=np.float64)
        cells = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.zeros((input_count,), dtype=np.int32)
        cell_constraints = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        cell_regions = np.zeros((cell_count,), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_mesh_copy_points"](handle, points.ctypes.data),
            "phx_mc_mesh_copy_points",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_cells"](handle, cells.ctypes.data),
            "phx_mc_mesh_copy_cells",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_vertex_map"](handle, vertex_map.ctypes.data),
            "phx_mc_mesh_copy_vertex_map",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_cell_constraints"](
                handle, cell_constraints.ctypes.data
            ),
            "phx_mc_mesh_copy_cell_constraints",
        )
        _raise_for_call(
            library["phx_mc_mesh_copy_cell_regions"](handle, cell_regions.ctypes.data),
            "phx_mc_mesh_copy_cell_regions",
        )
    finally:
        library["phx_mc_mesh_free"](handle)
    return NativeMeshArrays(
        points=points,
        cells=cells,
        vertex_map=vertex_map,
        cell_constraints=cell_constraints,
        cell_regions=cell_regions,
    )


def _limit(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    return bounded_integer(value, name, lower=1, upper=2**63 - 1)


def _triangulate(
    name: str,
    points: object,
    weights: object | None,
    width: int,
    max_cells: int | None,
    /,
    *,
    max_cavity_cells: int | None = None,
    max_work: int | None = None,
    max_scratch_bytes: int | None = None,
    record_native_resources: Callable[[PlanarExecutionEvidence], None] | None = None,
    admit_host_output: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    library = load_meshcore()
    point_array = _points(points, "points", width)
    count = point_array.shape[0]
    if count > _MAX_POINTS:
        raise ValueError(
            f"_meshcore.triangulation.point_count must be at most {_MAX_POINTS}; got {count}."
        )
    # Worst-case simplex counts: 2n - 5 triangles; (n^2 - 3n - 2) / 2 tetrahedra.
    default = (
        max(1, 2 * count) if width == 2 else max(1, (count * count - 3 * count) // 2)
    )
    limit = (
        _budget(max_cells, "max_cells")
        if width == 2
        else _limit(max_cells, default, "max_cells")
    )
    handle = ctypes.c_void_p()
    arguments: list[object] = [count, point_array.ctypes.data]
    if weights is not None:
        weight_array = _real_array(weights, "weights")
        if weight_array.shape != (count,):
            raise ValueError(
                f"_meshcore._triangulate.weights must have shape (n,); got {_input_summary(weights)}."
            )
        arguments.append(weight_array.ctypes.data)
    arguments.append(limit)
    work = np.empty((9,), dtype=np.uint64) if width == 2 else None
    memory = np.empty((6,), dtype=np.uint64) if width == 2 else None
    if width == 2:
        arguments.extend(
            (
                _budget(max_cavity_cells, "max_cavity_cells"),
                _budget(max_work, "max_work"),
                _budget(max_scratch_bytes, "max_scratch_bytes"),
                _pointer(work),
                _pointer(memory),
            )
        )
    arguments.append(ctypes.addressof(handle))
    status = library[name](*arguments)
    evidence = (
        PlanarExecutionEvidence(
            work, NativeExecutionMemoryCounters.from_slots(memory), MeshcoreStatus(status)
        )
        if work is not None and memory is not None
        else None
    )
    operation = name.removeprefix("phx_mc_")
    if status != MeshcoreStatus.OK:
        if evidence is not None and record_native_resources is not None:
            record_native_resources(evidence)
        if evidence is not None and status in (
            MeshcoreStatus.CAPACITY_EXCEEDED,
            MeshcoreStatus.INTERNAL_ERROR,
        ):
            raise NativeResourceFailure(evidence, operation)
        _raise_for_call(status, operation)
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._triangulate.name must satisfy the input contract ({operation}: missing mesh); got {_input_summary(name)}.",
        )
    mesh = _read_mesh(library, handle, admit_host_output=admit_host_output)
    cells = mesh.cells
    vertex_map = mesh.vertex_map
    if evidence is not None and record_native_resources is not None:
        record_native_resources(evidence)
    return cells, vertex_map


def delaunay_2d(
    points: object,
    /,
    *,
    max_triangles: int | None = None,
    max_cavity_cells: int | None = None,
    max_work: int | None = None,
    max_scratch_bytes: int | None = None,
    record_native_resources: Callable[[PlanarExecutionEvidence], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation of ``(n, 2)`` points.

    Returns ``(triangles (T, 3) int32, vertex_map (n,) int32)``; triangles are
    counterclockwise, canonically ordered, and cocircular ties are resolved by
    index-ordered symbolic perturbation. ``vertex_map`` sends duplicate points
    to their smallest identical index.

    All limits default to signed-int64 maximum; zero is a hard bound.
    ``record_native_resources`` receives work and allocation evidence,
    including resource refusals retained by :class:`NativeResourceFailure`.
    """

    return _triangulate(
        "phx_mc_delaunay_2d",
        points,
        None,
        2,
        max_triangles,
        max_cavity_cells=max_cavity_cells,
        max_work=max_work,
        max_scratch_bytes=max_scratch_bytes,
        record_native_resources=record_native_resources,
    )


def delaunay_3d(
    points: object, /, *, max_tetrahedra: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay tetrahedralization of ``(n, 3)`` points.

    Returns ``(tetrahedra (T, 4) int32, vertex_map (n,) int32)`` with positively
    oriented, canonically ordered tetrahedra.
    """

    return _triangulate("phx_mc_delaunay_3d", points, None, 3, max_tetrahedra)


def regular_2d(
    points: object,
    weights: object,
    /,
    *,
    max_triangles: int | None = None,
    max_cavity_cells: int | None = None,
    max_work: int | None = None,
    max_scratch_bytes: int | None = None,
    record_native_resources: Callable[[PlanarExecutionEvidence], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular (weighted Delaunay) triangulation; redundant points map to -1."""

    return _triangulate(
        "phx_mc_regular_2d",
        points,
        weights,
        2,
        max_triangles,
        max_cavity_cells=max_cavity_cells,
        max_work=max_work,
        max_scratch_bytes=max_scratch_bytes,
        record_native_resources=record_native_resources,
    )


def regular_3d(
    points: object,
    weights: object,
    /,
    *,
    max_tetrahedra: int | None = None,
    admit_host_output: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Regular (weighted Delaunay) tetrahedralization; redundant points map to -1."""

    return _triangulate(
        "phx_mc_regular_3d",
        points,
        weights,
        3,
        max_tetrahedra,
        admit_host_output=admit_host_output,
    )


def constrained_delaunay_2d(
    points: object,
    segments: object,
    /,
    *,
    holes: object | None = None,
    keep_convex_hull: bool = False,
    min_angle: float = 0.0,
    max_area: float = float("inf"),
    max_edge_length: float = float("inf"),
    lattice_spacing: float = 0.0,
    max_steiner: int = 0,
    max_triangles: int | None = None,
    max_cavity_cells: int | None = None,
    max_work: int | None = None,
    max_scratch_bytes: int | None = None,
    record_native_resources: Callable[[PlanarExecutionEvidence], None] | None = None,
    record_constraint_witnesses: Callable[[np.ndarray, np.ndarray, np.ndarray], None]
    | None = None,
) -> ConstrainedDelaunay2D:
    """Constrained Delaunay triangulation with Ruppert/Chew refinement.

    ``segments`` are ``(m, 2)`` point indices. Without ``keep_convex_hull`` the
    triangles outside the segment-bounded region and inside hole seeds are
    removed. ``min_angle`` (degrees, ``[0, 60)``), ``max_area``,
    ``max_edge_length`` (binary64 length of edges that are not input segment
    pieces; ``inf`` disables it) and ``max_steiner`` bound the refinement.
    ``lattice_spacing`` ``s > 0`` (``0`` disables it) first seeds the carved
    region with the sites of the equilateral lattice ``o + i (s, 0) +
    j (s/2, s sqrt(3)/2)`` (``o`` the lower input bounding-box corner), coarse
    to fine by dyadic sublattice level, that lie in the domain off its vertices
    and constrained edges, encroach no constrained subsegment of their
    constrained Delaunay cavity and stay at least ``s/2`` from every input point
    joined to them; seeds are free Steiner points within ``max_steiner``.
    Returns :class:`ConstrainedDelaunay2D` with ``points (N, 2), triangles
    (T, 3), vertex_map (input_count,), segment_ids (T, 3), status``.
    ``vertex_map`` explicitly maps authored input identities to output vertices;
    ``segment_ids[t, k]`` is the input segment carrying the edge opposite vertex
    ``k`` (or -1), and ``status`` is ``OK`` or ``REFINEMENT_LIMIT`` (valid mesh,
    quality targets unmet within ``max_steiner``).

    ``max_cavity_cells``, ``max_work`` and ``max_scratch_bytes`` apply before
    allocation and throughout preparation, insertion and refinement. All
    limits default to signed-int64 maximum; zero is a hard bound. The optional
    resource observer receives readonly :class:`PlanarExecutionEvidence`
    without changing the result arrays.

    ``record_constraint_witnesses`` receives readonly immediate parents (N, 2),
    input segment IDs (N,), and construction ratios (N,). Initial and free
    points use (-1, -1), -1, and zero; hull-only splits use segment -2.
    Interpret each binary64 ratio exactly and compose its endpoint witnesses.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 2)
    segment_array = _index_array(segments, "segments")
    if segment_array.ndim != 2 or segment_array.shape[1] != 2:
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.segments must have shape (m, 2); got {_input_summary(segments)}."
        )
    if np.any(segment_array < 0) or np.any(segment_array >= point_array.shape[0]):
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.segments must satisfy the input contract (segments reference a missing point.); got {_input_summary(segments)}."
        )
    segment_int = np.ascontiguousarray(segment_array, dtype=np.int32)
    hole_array = (
        np.zeros((0, 2), dtype=np.float64)
        if holes is None
        else _points(holes, "holes", 2)
    )
    if not isinstance(keep_convex_hull, bool):
        raise TypeError(
            f"_meshcore.constrained_delaunay_2d.keep_convex_hull must be a bool; got {_input_summary(keep_convex_hull)}."
        )
    angle = float(min_angle)
    area = float(max_area)
    edge_length = float(max_edge_length)
    spacing = float(lattice_spacing)
    if not (0.0 <= angle < 60.0):
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.min_angle must lie in [0, 60) degrees; got {_input_summary(min_angle)}."
        )
    if not area > 0.0:
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.max_area must be positive (inf disables the bound); got {_input_summary(max_area)}."
        )
    if not edge_length > 0.0:
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.max_edge_length must be positive (inf disables the bound); got {_input_summary(max_edge_length)}."
        )
    if not (spacing == 0.0 or (spacing > 0.0 and math.isfinite(spacing))):
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.lattice_spacing must be zero (disabled) or positive and finite; got {_input_summary(lattice_spacing)}."
        )
    if isinstance(max_steiner, bool) or not isinstance(max_steiner, (int, np.integer)):
        raise TypeError(
            f"_meshcore.constrained_delaunay_2d.max_steiner must be an integer; got {_input_summary(max_steiner)}."
        )
    if max_steiner < 0:
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.max_steiner must be nonnegative; got {_input_summary(max_steiner)}."
        )
    total = point_array.shape[0] + int(max_steiner)
    if total > _MAX_POINTS:
        raise ValueError(
            f"_meshcore.constrained_delaunay_2d.point_count must be at most {_MAX_POINTS}; got {total}."
        )
    limit = _budget(max_triangles, "max_triangles")
    work = np.empty((9,), dtype=np.uint64)
    memory = np.empty((6,), dtype=np.uint64)
    witness_capacity = total if record_constraint_witnesses is not None else 0
    witness_parents = np.empty((witness_capacity, 2), dtype=np.int32)
    witness_segments = np.empty((witness_capacity,), dtype=np.int32)
    witness_ratios = np.empty((witness_capacity,), dtype=np.float64)
    handle = ctypes.c_void_p()
    status = library["phx_mc_constrained_delaunay_2d"](
        ctypes.byref(
            _ConstrainedDelaunay2DRequest(
                point_count=point_array.shape[0],
                points=point_array.ctypes.data,
                segment_count=segment_int.shape[0],
                segments=segment_int.ctypes.data,
                hole_count=hole_array.shape[0],
                holes=hole_array.ctypes.data,
                keep_convex_hull=1 if keep_convex_hull else 0,
                min_angle_degrees=angle,
                max_area=area,
                max_edge_length=edge_length,
                lattice_spacing=spacing,
                max_steiner=int(max_steiner),
                max_triangles=limit,
                max_cavity_cells=_budget(max_cavity_cells, "max_cavity_cells"),
                max_work=_budget(max_work, "max_work"),
                max_scratch_bytes=_budget(max_scratch_bytes, "max_scratch_bytes"),
                work_evidence=work.ctypes.data,
                memory_evidence=memory.ctypes.data,
                witness_capacity=witness_capacity,
                witness_parents=witness_parents.ctypes.data if witness_capacity else None,
                witness_segments=witness_segments.ctypes.data
                if witness_capacity
                else None,
                witness_ratios=witness_ratios.ctypes.data if witness_capacity else None,
                mesh=ctypes.addressof(handle),
            )
        )
    )
    evidence = PlanarExecutionEvidence(
        work, NativeExecutionMemoryCounters.from_slots(memory), MeshcoreStatus(status)
    )
    code = MeshcoreStatus(status)
    if code not in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT):
        if record_native_resources is not None:
            record_native_resources(evidence)
        if code in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.INTERNAL_ERROR):
            raise NativeResourceFailure(evidence, "constrained_delaunay_2d")
        _raise_for_call(status, "constrained_delaunay_2d")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.constrained_delaunay_2d.points must satisfy the input contract (constrained_delaunay_2d: missing mesh); got {_input_summary(points)}.",
        )
    mesh = _read_mesh(library, handle)
    mesh_points = mesh.points
    cells = mesh.cells
    vertex_map = mesh.vertex_map
    cell_segments = mesh.cell_constraints
    if record_native_resources is not None:
        record_native_resources(evidence)
    if record_constraint_witnesses is not None:
        count = mesh_points.shape[0]
        parents = witness_parents[:count]
        sources = witness_segments[:count]
        ratios = witness_ratios[:count]
        for array in (parents, sources, ratios):
            array.setflags(write=False)
        record_constraint_witnesses(parents, sources, ratios)
    return ConstrainedDelaunay2D(
        points=mesh_points,
        triangles=cells,
        vertex_map=vertex_map,
        segment_ids=cell_segments,
        status=code,
    )


def _free_triangulation(library: MeshcoreLibrary, handle: int, /) -> None:
    library["phx_mc_triangulation_3d_free"](handle)


@final
class IncrementalDelaunay3D:
    """Owned native Delaunay tetrahedralization for incremental construction.

    The initial ``(n, 3)`` points are triangulated exactly as
    :func:`delaunay_3d`; later batches are inserted without rebuilding the hull.
    Every submitted point receives the next vertex id (initial points keep their
    indices) and the vertex map sends it to itself (a vertex), to the id of an
    earlier identical point, or to -1 (a refused insertion). The result depends
    only on the vertex set and ids, never on insertion order. ``max_vertices``
    bounds the ids, ``max_tetrahedra`` the finite cells and ``max_cavity`` the
    tetrahedra one insertion may remove; every refusal is decided before any
    change, so refused insertions leave the triangulation unchanged. The
    native state is released by :meth:`close` or when the object is collected.
    """

    __slots__ = ("__weakref__", "_finalizer", "_handle", "_library")

    def __init__(
        self,
        points: object,
        /,
        *,
        max_vertices: int | None = None,
        max_tetrahedra: int | None = None,
        max_cavity: int | None = None,
    ) -> None:
        library = load_meshcore()
        point_array = _points(points, "points", 3)
        count = point_array.shape[0]
        vertices = _limit(max_vertices, max(count, 1), "max_vertices")
        if vertices < count or vertices > _MAX_POINTS:
            raise ValueError(
                f"IncrementalDelaunay3D.max_vertices must lie in [{count}, {_MAX_POINTS}]; got {vertices}."
            )
        # Worst-case tetrahedron count of max_vertices points (see delaunay_3d).
        default = max(1, (vertices * vertices - 3 * vertices) // 2)
        tetrahedra = _limit(max_tetrahedra, default, "max_tetrahedra")
        cavity = _limit(max_cavity, 2**31 - 1, "max_cavity")
        handle = ctypes.c_void_p()
        status = library["phx_mc_triangulation_3d_create"](
            count,
            point_array.ctypes.data,
            vertices,
            tetrahedra,
            cavity,
            ctypes.addressof(handle),
        )
        _raise_for_call(status, "triangulation_3d_create")
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.IncrementalDelaunay3D.__init__.points must satisfy the input contract (triangulation_3d_create: missing handle); got {_input_summary(points)}.",
            )
        self._library = library
        self._handle = handle.value
        self._finalizer = weakref.finalize(
            self, _free_triangulation, library, handle.value
        )

    def _live(self) -> int:
        if not self._finalizer.alive:
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D._live.self must satisfy the input contract (the incremental triangulation is closed.); got {_input_summary(self)}."
            )
        return self._handle

    def close(self) -> None:
        """Release the native state; later calls raise ``ValueError``."""

        self._finalizer()

    def insert(
        self, points: object, /, *, work_limit: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Insert a batch of ``(m, 3)`` points.

        Returns ``(vertices (m,) int32, status (m,) int32)``: the vertex map of
        each point and ``OK``, ``CAPACITY_EXCEEDED`` (work, cavity, cell or slot
        limit; after the ``work_limit`` of this call is exhausted the remaining
        points are refused) or ``CONSTRAINT_INTERSECTION`` (the conflict region
        bounded by constrained facets is not star-shaped from the point, e.g. a
        point on a constrained facet). Non-finite or out-of-domain points, or
        more ids than ``max_vertices``, refuse the whole batch unchanged.
        """

        handle = self._live()
        point_array = _points(points, "points", 3)
        budget = _limit(work_limit, 2**63 - 1, "work_limit")
        count = point_array.shape[0]
        vertices = np.zeros((count,), dtype=np.int32)
        status = np.zeros((count,), dtype=np.int32)
        call = self._library["phx_mc_triangulation_3d_insert"](
            handle,
            count,
            point_array.ctypes.data,
            budget,
            vertices.ctypes.data,
            status.ctypes.data,
        )
        _raise_for_call(call, "triangulation_3d_insert")
        return vertices, status

    def constrain_facets(self, facets: object, constraint_ids: object, /) -> np.ndarray:
        """Mark existing facets ``(m, 3)`` (vertex ids) with references ``(m,) >= 0``.

        Returns the item status: ``OK`` (also for an equal existing reference)
        or ``INVALID_INPUT`` for a facet absent from the triangulation, dead or
        repeated ids, or a different existing reference. Later insertions never
        cross a constrained facet.
        """

        handle = self._live()
        facet_array = _index_array(facets, "facets")
        id_array = _index_array(constraint_ids, "constraint_ids")
        if facet_array.ndim != 2 or facet_array.shape[1] != 3:
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D.constrain_facets.facets must have shape (m, 3); got {_input_summary(facets)}."
            )
        if id_array.shape != facet_array.shape[:1]:
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D.constrain_facets.constraint_ids must have shape (m,); got {_input_summary(constraint_ids)}."
            )
        if np.any(np.abs(facet_array) > _MAX_POINTS) or np.any(
            np.abs(id_array) > 2**31 - 1
        ):
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D.constrain_facets.facets must fit int32; got {_input_summary(facets)}."
            )
        facet_int = np.ascontiguousarray(facet_array, dtype=np.int32)
        id_int = np.ascontiguousarray(id_array, dtype=np.int32)
        count = facet_int.shape[0]
        status = np.zeros((count,), dtype=np.int32)
        call = self._library["phx_mc_triangulation_3d_constrain_facets"](
            handle, count, facet_int.ctypes.data, id_int.ctypes.data, status.ctypes.data
        )
        _raise_for_call(call, "triangulation_3d_constrain_facets")
        return status

    def label_regions(self, seeds: object, labels: object, /) -> np.ndarray:
        """Relabel regions from seeds ``(s, 3)`` with labels ``(s,) >= 0``.

        Clears every region, then labels the finite cells reachable from the cell
        strictly containing each seed without crossing a constrained facet.
        Returns the item status: ``INVALID_INPUT`` for a seed outside the hull,
        on a facet, edge or vertex, with a negative label, or reaching a region
        labeled differently. Cells created later inherit the region of the
        cells they replace.
        """

        handle = self._live()
        seed_array = _points(seeds, "seeds", 3)
        label_array = _index_array(labels, "labels")
        if label_array.shape != seed_array.shape[:1]:
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D.label_regions.labels must have shape (s,); got {_input_summary(labels)}."
            )
        if np.any(np.abs(label_array) > 2**31 - 1):
            raise ValueError(
                f"_meshcore.IncrementalDelaunay3D.label_regions.labels must fit int32; got {_input_summary(labels)}."
            )
        label_int = np.ascontiguousarray(label_array, dtype=np.int32)
        count = seed_array.shape[0]
        status = np.zeros((count,), dtype=np.int32)
        call = self._library["phx_mc_triangulation_3d_label_regions"](
            handle,
            count,
            seed_array.ctypes.data,
            label_int.ctypes.data,
            status.ctypes.data,
        )
        _raise_for_call(call, "triangulation_3d_label_regions")
        return status

    def locate(self, points: object, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Location of ``(m, 3)`` points.

        Returns ``(cells (m, 4) int32, locations (m,) int8, status (m,) int32)``:
        the vertex tuple (even permutation starting at its smallest
        entry, -1 for the ghost vertex) of a tetrahedron containing each point,
        and the number of its facet planes through the point (0 interior,
        1 facet, 2 edge, 3 vertex) or -1 outside the convex hull.
        """

        handle = self._live()
        point_array = _points(points, "points", 3)
        count = point_array.shape[0]
        cells = np.zeros((count, 4), dtype=np.int32)
        locations = np.zeros((count,), dtype=np.int8)
        status = np.zeros((count,), dtype=np.int32)
        call = self._library["phx_mc_triangulation_3d_locate"](
            handle,
            count,
            point_array.ctypes.data,
            cells.ctypes.data,
            locations.ctypes.data,
            status.ctypes.data,
        )
        _raise_for_call(call, "triangulation_3d_locate")
        return cells, locations, status

    def statistics(self) -> Triangulation3DStatistics:
        """Int64 counters named by :class:`Triangulation3DStatistics`."""

        handle = self._live()
        values = np.zeros((13,), dtype=np.int64)
        call = self._library["phx_mc_triangulation_3d_statistics"](
            handle, values.ctypes.data
        )
        _raise_for_call(call, "triangulation_3d_statistics")
        return Triangulation3DStatistics.from_slots(values)

    def finalize(self) -> NativeMeshArrays:
        """Snapshot; the triangulation stays usable.

        Returns :class:`NativeMeshArrays` with ``points (N, 3), tetrahedra (T, 4), vertex_map (N,),
        cell_constraints (T, 4), cell_regions (T,)``: every submitted point, the
        positively oriented canonically ordered finite cells, the vertex map,
        the constraint reference of the facet opposite each cell vertex (-1
        when unconstrained) and each cell's region (-1 when unlabeled).
        """

        handle = self._live()
        mesh = ctypes.c_void_p()
        status = self._library["phx_mc_triangulation_3d_finalize"](
            handle, ctypes.addressof(mesh)
        )
        _raise_for_call(status, "triangulation_3d_finalize")
        if not mesh.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.IncrementalDelaunay3D.finalize.self must satisfy the input contract (triangulation_3d_finalize: missing mesh); got {_input_summary(self)}.",
            )
        return _read_mesh(self._library, mesh)


# ---------------------------------------------------------------- graph partition


def graph_partition(
    offsets: np.ndarray,
    neighbors: np.ndarray,
    edge_weights: np.ndarray,
    vertex_weights: np.ndarray,
    part_targets: np.ndarray,
    part_capacities: np.ndarray,
    /,
    *,
    require_nonempty: bool,
    refinement_passes: int,
    work_limit: int,
) -> GraphPartition:
    """Multilevel k-way partition of a symmetric weighted CSR graph.

    Returns :class:`GraphPartition` with ``parts, counters``: int32 part per vertex and the int64 work
    counters named by :class:`GraphPartitionCounters`. The semantic owner,
    :mod:`phydrax.graph`, validates the graph and part table first; the native
    call revalidates them and reports a malformed one as ``ValueError``. An
    exhausted ``work_limit`` raises :class:`MeshcoreError` with status
    ``CAPACITY_EXCEEDED`` and the adjacency visits plus candidate evaluations reached.
    """

    library = load_meshcore()
    xadj = np.ascontiguousarray(offsets, dtype=np.int64)
    adjacency = np.ascontiguousarray(neighbors, dtype=np.int32)
    edges = np.ascontiguousarray(edge_weights, dtype=np.int64)
    vertices = np.ascontiguousarray(vertex_weights, dtype=np.int64)
    targets = np.ascontiguousarray(part_targets, dtype=np.int64)
    capacities = np.ascontiguousarray(part_capacities, dtype=np.int64)
    count = vertices.shape[0]
    if (
        xadj.shape != (count + 1,)
        or adjacency.shape != edges.shape
        or targets.shape != capacities.shape
        or targets.ndim != 1
    ):
        raise ValueError(
            f"_meshcore.graph_partition.offsets must satisfy the input contract (graph_partition: misaligned CSR or part arrays.); got {_input_summary(offsets)}."
        )
    parts = np.zeros((count,), dtype=np.int32)
    counters = np.zeros((11,), dtype=np.int64)
    status = library["phx_mc_graph_partition"](
        count,
        xadj.ctypes.data,
        adjacency.ctypes.data,
        edges.ctypes.data,
        vertices.ctypes.data,
        targets.shape[0],
        targets.ctypes.data,
        capacities.ctypes.data,
        1 if require_nonempty else 0,
        refinement_passes,
        work_limit,
        parts.ctypes.data,
        counters.ctypes.data,
    )
    if status == MeshcoreStatus.CAPACITY_EXCEEDED:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"graph_partition.work_units must not exceed {work_limit}; "
            f"got {counters[9]} adjacency visits and {counters[10]} candidate evaluations.",
        )
    _raise_for_call(status, "graph_partition")
    return GraphPartition(
        parts=parts, counters=GraphPartitionCounters.from_slots(counters)
    )


# ---------------------------------------------------------------- surface reconnection


def _surface_metric_buffer(
    value: object | None, count: int, name: str, /
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = _real_array(value, name)
    if array.shape != (count, 3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{'_meshcore.' + name} must contain one finite ambient three-dimensional tensor per vertex; got {_input_summary(value)}."
        )
    return array


def _surface_pole_buffer(
    value: object | None, count: int, name: str, /
) -> NDArray[np.int64] | None:
    if value is None:
        return None
    array = _index_array(value, name)
    if (
        array.shape != (count,)
        or np.any(array < -1)
        or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            f"{'_meshcore.' + name} must contain signed-int64 pole IDs or -1 for ordinary vertices; got {_input_summary(value)}."
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def surface_reconnect(
    charts: ArrayLike,
    points: ArrayLike,
    normals: ArrayLike,
    triangles: ArrayLike,
    constrained: ArrayLike,
    insert_charts: ArrayLike,
    insert_points: ArrayLike,
    insert_normals: ArrayLike,
    insert_spacing: ArrayLike,
    insert_hints: ArrayLike,
    /,
    *,
    sweep: bool,
    max_triangles: int,
    work_limit: int,
    vertex_metrics: ArrayLike | None = None,
    insert_metrics: ArrayLike | None = None,
    vertex_pole_ids: ArrayLike | None = None,
    insert_pole_ids: ArrayLike | None = None,
    insert_edges: ArrayLike | None = None,
) -> SurfaceReconnection:
    """Insert chart points into one chart-embedded surface patch and reconnect it.

    ``charts`` ``(n, 2)``, ``points`` ``(n, 3)`` and ``normals`` ``(n, 3)`` are
    the chart coordinates, physical coordinates and oriented source normals
    (zero where undefined, e.g. at a pole) of the vertices, ``triangles``
    ``(m, 3)`` are counterclockwise in the chart and ``constrained`` ``(m, 3)``
    flags the edge opposite each vertex. Points ``insert_charts`` ``(k, 2)`` /
    ``insert_points`` ``(k, 3)`` / ``insert_normals`` ``(k, 3)`` are inserted in
    order (refused within ``insert_spacing`` of an existing vertex, on a
    constrained edge, outside the chart domain, or when splitting a triangle
    consistent with the source normals would make it inconsistent), each
    followed by physical Delaunay flips that never degrade that consistency;
    ``sweep`` flips every edge afterwards. Returns :class:`SurfaceReconnection` with ``triangles, constrained,
    vertex_ids, item_status, counters``: accepted points receive ids ``n,
    n + 1, ...`` (``-1`` when refused), ``item_status`` holds
    :class:`MeshcoreStatus` codes and ``counters`` the values named by
    :class:`SurfaceReconnectionCounters`. An invalid chart triangulation raises
    ``ValueError``.
    ``insert_edges`` optionally selects existing unconstrained reciprocal edges
    by local source vertex rows, with shape ``(k, 2)``; ``(-1, -1)`` requests
    ordinary point insertion. A refused selected edge is never reinterpreted
    as a generic point insertion. With ``sweep=False``, a noncollinear rounded
    point is the execution view of a producer-owned rational edge
    restriction. A declared radial pole complex may defer physical-child
    validation to the producer, which must validate the chart and
    continuous source before publication or another reconnect.
    """

    library = load_meshcore()
    chart_array = _points(charts, "charts", 2)
    point_array = _points(points, "points", 3)
    normal_array = _points(normals, "normals", 3)
    cells = _power_indices(triangles, "triangles")
    flags = np.ascontiguousarray(np.asarray(constrained, dtype=np.bool_), dtype=np.int8)
    new_charts = _points(insert_charts, "insert_charts", 2)
    new_points = _points(insert_points, "insert_points", 3)
    new_normals = _points(insert_normals, "insert_normals", 3)
    spacing = np.ascontiguousarray(
        _real_array(insert_spacing, "insert_spacing"), dtype=np.float64
    )
    hints = _power_indices(insert_hints, "insert_hints")
    count = new_charts.shape[0]
    if (
        point_array.shape[0] != chart_array.shape[0]
        or normal_array.shape != point_array.shape
        or cells.ndim != 2
        or cells.shape[1] != 3
        or flags.shape != cells.shape
        or new_points.shape[0] != count
        or new_normals.shape[0] != count
        or spacing.shape != (count,)
        or hints.shape != (count,)
    ):
        raise ValueError(
            f"_meshcore.surface_reconnect.charts must satisfy the input contract (surface_reconnect: misaligned vertex, cell or insertion arrays.); got {_input_summary(charts)}."
        )
    edges = None
    if insert_edges is not None:
        edge_values = np.asarray(insert_edges)
        if edge_values.shape != (count, 2):
            raise ValueError(
                f"_meshcore.surface_reconnect.insert_edges must have shape (insert_count, 2); got {_input_summary(insert_edges)}."
            )
        if edge_values.dtype.kind not in "iu":
            raise TypeError(
                f"_meshcore.surface_reconnect.insert_edges must contain integer vertex rows; got {_input_summary(insert_edges)}."
            )
        if np.any(edge_values < -1) or np.any(edge_values >= point_array.shape[0]):
            raise ValueError(
                f"_meshcore.surface_reconnect.insert_edges must satisfy the input contract (insert_edges vertex rows are outside the source vertex range.); got {_input_summary(insert_edges)}."
            )
        generic = np.all(edge_values == -1, axis=1)
        if np.any(np.any(edge_values == -1, axis=1) & ~generic):
            raise ValueError(
                f"_meshcore.surface_reconnect.insert_edges must be paired (-1, -1); got {_input_summary(insert_edges)}."
            )
        if np.any((edge_values[:, 0] == edge_values[:, 1]) & ~generic):
            raise ValueError(
                f"_meshcore.surface_reconnect.insert_edges must have distinct selected endpoints; got {_input_summary(insert_edges)}."
            )
        if np.any(edge_values > np.iinfo(np.int32).max):
            raise ValueError(
                f"_meshcore.surface_reconnect.insert_edges must fit int32; got {_input_summary(insert_edges)}."
            )
        if not np.all(generic):
            edges = np.ascontiguousarray(edge_values, dtype=np.int32)
    metrics = _surface_metric_buffer(
        vertex_metrics, point_array.shape[0], "vertex_metrics"
    )
    new_metrics = _surface_metric_buffer(insert_metrics, count, "insert_metrics")
    if metrics is not None and new_metrics is None and count == 0:
        new_metrics = np.empty((0, 3, 3), dtype=np.float64)
    if (metrics is None) != (new_metrics is None):
        raise ValueError(
            f"_meshcore.surface_reconnect.vertex_metrics must be supplied together; got {_input_summary(vertex_metrics)}."
        )
    poles = _surface_pole_buffer(vertex_pole_ids, point_array.shape[0], "vertex_pole_ids")
    new_poles = _surface_pole_buffer(insert_pole_ids, count, "insert_pole_ids")
    capacity = _limit(max_triangles, cells.shape[0] + 2 * count, "max_triangles")
    triangles_out = np.zeros((capacity, 3), dtype=np.int32)
    flags_out = np.zeros((capacity, 3), dtype=np.int8)
    produced = np.zeros((1,), dtype=np.int64)
    vertex_ids = np.full((count,), -1, dtype=np.int32)
    item_status = np.zeros((count,), dtype=np.int32)
    counters = np.zeros((6,), dtype=np.int64)
    status = library["phx_mc_surface_reconnect"](
        ctypes.byref(
            _SurfaceReconnectRequest(
                vertex_count=chart_array.shape[0],
                charts=chart_array.ctypes.data,
                points=point_array.ctypes.data,
                normals=normal_array.ctypes.data,
                vertex_metrics=_pointer(metrics),
                vertex_pole_ids=_pointer(poles),
                triangle_count=cells.shape[0],
                triangles=cells.ctypes.data,
                constrained=flags.ctypes.data,
                insert_count=count,
                insert_charts=new_charts.ctypes.data,
                insert_points=new_points.ctypes.data,
                insert_normals=new_normals.ctypes.data,
                insert_metrics=_pointer(new_metrics),
                insert_pole_ids=_pointer(new_poles),
                insert_spacing=spacing.ctypes.data,
                insert_hints=hints.ctypes.data,
                insert_edges=_pointer(edges),
                sweep=_boolean_flag(sweep, "sweep"),
                max_triangles=capacity,
                work_limit=_budget(work_limit, "work_limit"),
                triangles_out=triangles_out.ctypes.data,
                constrained_out=flags_out.ctypes.data,
                triangle_count_out=produced.ctypes.data,
                vertex_ids=vertex_ids.ctypes.data,
                item_status=item_status.ctypes.data,
                counters=counters.ctypes.data,
            )
        )
    )
    _raise_for_call(status, "surface_reconnect")
    rows = int(produced[0])
    return SurfaceReconnection(
        triangles=triangles_out[:rows],
        constrained=flags_out[:rows].astype(np.bool_),
        vertex_ids=vertex_ids,
        item_status=item_status,
        counters=SurfaceReconnectionCounters.from_slots(counters),
    )


def periodic_delaunay(
    points: np.ndarray,
    fractional: np.ndarray,
    lattice: np.ndarray,
    inverse: np.ndarray,
    /,
    *,
    initial_margin: float,
    max_images: int,
    max_cells: int,
) -> PeriodicDelaunay:
    """Periodic Delaunay triangulation on certified bounded image neighborhoods.

    ``points`` (n, d) are representatives with unwrapped lattice coordinates
    ``fractional`` (finite, magnitude at most 2**30); each representative's base
    image is wrapped into [0, 1) by the integer shift ``-floor(fractional)``.
    ``lattice`` (d, d) holds lattice vectors as rows and ``inverse`` (d, d) maps
    Cartesian to fractional coordinates. Returns :class:`PeriodicDelaunay` with
    ``status, vertices (T, d + 1) int32, shifts (T, d + 1, d) int32, evidence)``
    with ``evidence`` named by :class:`PeriodicDelaunayEvidence`. The domain
    outcomes ``OK``, ``CAPACITY_EXCEEDED`` (image or cell budget, see
    ``exhausted_limit``) and ``INVALID_INPUT`` (a repeated orbit) are returned
    with their evidence for the semantic owner to report; every other status,
    including out-of-range fractional coordinates, raises.
    """

    library = load_meshcore()
    point_array = _real_array(points, "points")
    if point_array.ndim != 2 or point_array.shape[1] not in (2, 3):
        raise ValueError(
            f"_meshcore.periodic_delaunay.points must have shape (n, 2) or (n, 3); got {_input_summary(points)}."
        )
    count, dimension = point_array.shape
    fractional_array = _points(fractional, "fractional", dimension)
    lattice_array = _real_array(lattice, "lattice")
    inverse_array = _real_array(inverse, "inverse")
    if (
        fractional_array.shape != point_array.shape
        or lattice_array.shape != (dimension, dimension)
        or inverse_array.shape != (dimension, dimension)
    ):
        raise ValueError(
            f"_meshcore.periodic_delaunay.lattice must satisfy the input contract (periodic_delaunay: misaligned point or lattice arrays.); got {_input_summary(lattice)}."
        )
    if count < 1 or count > _MAX_POINTS:
        raise ValueError(
            f"_meshcore.periodic_delaunay.points must satisfy the input contract (periodic_delaunay requires between 1 and 2**31 - 2 points.); got {_input_summary(points)}."
        )
    evidence = np.zeros((10,), dtype=np.float64)
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_periodic_delaunay"](
            dimension,
            count,
            point_array.ctypes.data,
            fractional_array.ctypes.data,
            lattice_array.ctypes.data,
            inverse_array.ctypes.data,
            float(initial_margin),
            _limit(max_images, 1, "max_images"),
            _limit(max_cells, 1, "max_cells"),
            evidence.ctypes.data,
            ctypes.addressof(handle),
        )
    )
    match status:
        case MeshcoreStatus.OK:
            pass
        case MeshcoreStatus.CAPACITY_EXCEEDED | MeshcoreStatus.INVALID_INPUT:
            return PeriodicDelaunay(
                status=status,
                vertices=np.zeros((0, dimension + 1), dtype=np.int32),
                shifts=np.zeros((0, dimension + 1, dimension), dtype=np.int32),
                evidence=PeriodicDelaunayEvidence.from_slots(evidence),
            )
        case _:
            _raise_for_call(status, "periodic_delaunay")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.periodic_delaunay.points must satisfy the input contract (periodic_delaunay: missing triangulation); got {_input_summary(points)}.",
        )
    # The native handle is released even when a copy fails.
    try:
        cells = library["phx_mc_periodic_cell_count"](handle)
        vertices = np.zeros((cells, dimension + 1), dtype=np.int32)
        shifts = np.zeros((cells, dimension + 1, dimension), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_periodic_copy_cells"](
                handle, vertices.ctypes.data, shifts.ctypes.data
            ),
            "phx_mc_periodic_copy_cells",
        )
    finally:
        library["phx_mc_periodic_free"](handle)
    return PeriodicDelaunay(
        status=status,
        vertices=vertices,
        shifts=shifts,
        evidence=PeriodicDelaunayEvidence.from_slots(evidence),
    )


def periodic_refine_2d(
    points: np.ndarray,
    lattice: np.ndarray,
    inverse: np.ndarray,
    vertices: np.ndarray,
    shifts: np.ndarray,
    controls: np.ndarray,
    quantiles: np.ndarray,
    /,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_vertices: int,
    max_cells: int,
    max_images: int,
    max_work_units: int,
    max_cavity_cells: int,
    max_scratch_bytes: int,
    protected_edges: np.ndarray | None = None,
) -> PeriodicRefinement:
    """Run one native incremental periodic frontal owner under the active scope.

    Control rows are target, optional minimum/maximum/growth (NaN when absent),
    and hard strength. Native outcomes retain deterministic counters on refusal.
    Protected rows bind quotient-edge identities including lattice shifts.
    """
    library = load_meshcore()
    point_array = _points(points, "periodic_refine.points", 2)
    lattice_array = _real_array(lattice, "periodic_refine.lattice")
    inverse_array = _real_array(inverse, "periodic_refine.inverse")
    vertex_array = _index_array(vertices, "periodic_refine.vertices")
    shift_array = _index_array(shifts, "periodic_refine.shifts")
    control_array = _points(controls, "periodic_refine.controls", 5)
    quantile_array = _real_array(quantiles, "periodic_refine.quantiles")
    protected_array = (
        np.empty((0, 4), dtype=np.int32)
        if protected_edges is None
        else _index_array(protected_edges, "periodic_refine.protected_edges")
    )
    if (
        lattice_array.shape != (2, 2)
        or inverse_array.shape != (2, 2)
        or vertex_array.ndim != 2
        or vertex_array.shape[1] != 3
        or shift_array.shape != (*vertex_array.shape, 2)
        or quantile_array.ndim != 1
        or protected_array.ndim != 2
        or protected_array.shape[1] != 4
    ):
        raise ValueError(
            "_meshcore.periodic_refine.arrays must bind a two-dimensional quotient; got incompatible shapes"
        )
    if (
        np.any(vertex_array < 0)
        or np.any(vertex_array >= point_array.shape[0])
        or np.any(shift_array < np.iinfo(np.int32).min)
        or np.any(shift_array > np.iinfo(np.int32).max)
        or np.any(protected_array[:, :2] < 0)
        or np.any(protected_array[:, :2] >= point_array.shape[0])
        or np.any(protected_array[:, 2:] < np.iinfo(np.int32).min)
        or np.any(protected_array[:, 2:] > np.iinfo(np.int32).max)
    ):
        raise ValueError(
            "_meshcore.periodic_refine.indices must identify admitted points and int32 shifts; got out-of-range indices"
        )
    vertex_array = np.ascontiguousarray(vertex_array, dtype=np.int32)
    shift_array = np.ascontiguousarray(shift_array, dtype=np.int32)
    protected_array = np.ascontiguousarray(protected_array, dtype=np.int32)
    request = _PeriodicRefinementInput(
        point_array.shape[0],
        point_array.ctypes.data,
        lattice_array.ctypes.data,
        inverse_array.ctypes.data,
        vertex_array.shape[0],
        vertex_array.ctypes.data,
        shift_array.ctypes.data,
        control_array.shape[0],
        control_array.ctypes.data,
        quantile_array.size,
        quantile_array.ctypes.data,
        float(absolute_tolerance),
        float(relative_tolerance),
        _limit(max_vertices, 1, "max_vertices"),
        _limit(max_cells, 1, "max_cells"),
        _limit(max_images, 1, "max_images"),
        _budget(max_work_units, "max_work_units"),
        _budget(max_cavity_cells, "max_cavity_cells"),
        _unsigned_budget(max_scratch_bytes, "max_scratch_bytes"),
        protected_array.shape[0],
        protected_array.ctypes.data,
    )
    slots = np.zeros(10, dtype=np.uint64)
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_periodic_refine_2d"](
            ctypes.byref(request), slots.ctypes.data, ctypes.addressof(handle)
        )
    )
    counters = PeriodicRefinementCounters(*(int(value) for value in slots))
    if status != MeshcoreStatus.OK:
        if status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.INVALID_INPUT):
            _raise_for_call(status, "periodic_refine_2d")
        return PeriodicRefinement(
            status,
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3, 2), dtype=np.int32),
            counters,
        )
    try:
        count = library["phx_mc_periodic_refinement_point_count"](handle)
        cells = library["phx_mc_periodic_refinement_cell_count"](handle)
        output = np.empty((count, 2), dtype=np.float64)
        output_vertices = np.empty((cells, 3), dtype=np.int32)
        output_shifts = np.empty((cells, 3, 2), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_periodic_refinement_copy_points"](handle, output.ctypes.data),
            "periodic_refinement_copy_points",
        )
        _raise_for_call(
            library["phx_mc_periodic_refinement_copy_cells"](
                handle, output_vertices.ctypes.data, output_shifts.ctypes.data
            ),
            "periodic_refinement_copy_cells",
        )
    finally:
        library["phx_mc_periodic_refinement_free"](handle)
    return PeriodicRefinement(status, output, output_vertices, output_shifts, counters)


# ---------------------------------------------------------------- PLC recovery

TetMeshBoundaryPolicy: TypeAlias = Literal["fixed", "conforming"]


PLC_3D_FAILURE_REASONS = (
    "none",
    "duplicate_point",
    "invalid_polygon",
    "invalid_facet",
    "intersecting_constraints",
    "open_boundary",
    "inconsistent_regions",
    "region_leak",
    "fixed_segment",
    "fixed_facet",
    "nonrepresentable_steiner",
    "vertex_budget",
    "tetrahedron_budget",
    "work_budget",
    "invalid_seed",
    "outside_domain",
    "scratch_byte_budget",
    "source_deviation",
)
PLC_3D_ENTITY_KINDS = ("none", "point", "polygon", "facet", "plc_edge", "region", "seed")


@final
class PlcRecoveryFailure(MeshcoreError):
    """Refused PLC recovery: reason, involved entities and work counters.

    ``entities`` holds up to two ``(kind, id)`` pairs named by
    :data:`PLC_3D_ENTITY_KINDS` (e.g. two intersecting polygons, a region and
    the edge where its boundary is open); ``counters`` follow
    :class:`Plc3DCounters`. ``source_refusal`` is the certified deviation
    and the declared bound of a ``"source_deviation"`` refusal (zeros
    otherwise).
    """

    def __init__(
        self,
        status: MeshcoreStatus,
        reason: str,
        entities: tuple[tuple[str, int], ...],
        counters: Plc3DCounters,
        /,
        *,
        source_refusal: tuple[float, float] = (0.0, 0.0),
        memory_counters: NativeExecutionMemoryCounters | None = None,
    ) -> None:
        described = ", ".join(f"{kind} {index}" for kind, index in entities)
        super().__init__(
            status,
            f"plc3d_recover: {reason} ({described})",
            memory_counters=memory_counters,
        )
        self.reason = reason
        self.entities = entities
        self.counters = counters
        self.source_refusal = source_refusal


@final
class PlcRecovery3D(StrictModule):
    """Recovered constrained tetrahedralization of a PLC (host arrays).

    ``points`` keep the input vertices first; Steiner points follow with
    ``vertex_dimension`` 1 (on a PLC edge), 2 (on a facet) or 3 (interior
    cavity-fill point). Domain
    tetrahedra are positively oriented with their region; ``faces`` are every
    constrained subfacet oriented like its source facet (``face_sources``);
    ``segments`` the subsegments of every PLC edge (``segment_sources`` index
    ``plc_edges``: explicit segments first, then facet-group boundary edges).
    ``protection_radii`` are the protecting-ball radii (0 unprotected);
    ``input_triangles`` are the triangulation of the input polygons
    (``input_polygons`` names each triangle's polygon). Every point carries a
    source construction record: stratum (0 none, 1 PLC edge, 2 input triangle), the
    ``plc_edges`` or ``input_triangles`` row, its parameters (a line
    parameter from the edge's first endpoint, or the weights of the
    triangle's second and third corners) and its certified outward deviation
    (zero exactly on the source; positive only for an ancestry-backed
    carrier admitted by the declared facet/segment tolerances).
    """

    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    protection_radii: np.ndarray
    plc_edges: np.ndarray
    vertex_dimension: np.ndarray
    input_triangles: np.ndarray
    input_polygons: np.ndarray
    witness_strata: NDArray[np.int8]
    witness_entities: NDArray[np.int32]
    witness_parameters: NDArray[np.float64]
    witness_deviations: NDArray[np.float64]
    counters: Plc3DCounters
    memory_counters: NativeExecutionMemoryCounters | None


@final
class PlcSourceConstraints(StrictModule):
    """Validated input PLC strata; no tetrahedral recovery is implied."""

    points: NDArray[np.float64]
    plc_edges: NDArray[np.int32]
    input_triangles: NDArray[np.int32]
    input_polygons: NDArray[np.int32]
    counters: Plc3DCounters
    memory_counters: NativeExecutionMemoryCounters | None

    def __post_init__(self) -> None:
        for values in (
            self.points,
            self.plc_edges,
            self.input_triangles,
            self.input_polygons,
        ):
            values.setflags(write=False)


def _plc_result_evidence(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    status: int,
    record_native_phase: NativePhaseRecorder | None,
    /,
) -> tuple[Plc3DCounters, NativeExecutionMemoryCounters | None]:
    memory_values = np.empty((6,), dtype=np.uint64)
    memory_enabled = np.empty((1,), dtype=np.int32)
    _raise_for_call(
        library["phx_mc_plc3d_memory_evidence"](
            handle,
            memory_values.ctypes.data,
            memory_enabled.ctypes.data,
        ),
        "plc3d_memory_evidence",
    )
    memory = (
        NativeExecutionMemoryCounters.from_slots(memory_values)
        if memory_enabled[0]
        else None
    )
    if record_native_phase is not None:
        elapsed = np.zeros((5,), dtype=np.int64)
        invocations = np.zeros((5,), dtype=np.int64)
        enabled = np.zeros((1,), dtype=np.int32)
        _raise_for_call(
            library["phx_mc_plc3d_phase_times"](
                handle,
                elapsed.ctypes.data,
                invocations.ctypes.data,
                enabled.ctypes.data,
            ),
            "plc3d_phase_times",
        )
        if not enabled[0] or np.any(elapsed < 0) or np.any(invocations < 0):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore._plc_result_evidence.library must satisfy the input contract (invalid PLC phase measurements); got {_input_summary(library)}.",
            )
        phases = (
            "validation",
            "preparation",
            "boundary_recovery",
            "classification",
            "publication",
        )
        for index, phase in enumerate(phases):
            if invocations[index]:
                record_native_phase(
                    phase,
                    float(elapsed[index]) * 1e-9,
                    None,
                    int(invocations[index]),
                )
    slots = np.zeros((14,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_plc3d_counters"](handle, slots.ctypes.data),
        "phx_mc_plc3d_counters",
    )
    counters = Plc3DCounters.from_slots(slots)
    if status != MeshcoreStatus.OK:
        failure = np.zeros((5,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_plc3d_failure"](handle, failure.ctypes.data),
            "phx_mc_plc3d_failure",
        )
        refusal = np.zeros((2,), dtype=np.float64)
        empty = np.empty((0,), dtype=np.float64)
        _raise_for_call(
            library["phx_mc_plc3d_source_witnesses"](
                handle,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                refusal.ctypes.data,
            ),
            "phx_mc_plc3d_source_witnesses",
        )
        entities = tuple(
            (PLC_3D_ENTITY_KINDS[int(kind)], int(index))
            for kind, index in ((failure[1], failure[2]), (failure[3], failure[4]))
            if kind != 0
        )
        raise PlcRecoveryFailure(
            MeshcoreStatus(status),
            PLC_3D_FAILURE_REASONS[int(failure[0])],
            entities,
            counters,
            source_refusal=(float(refusal[0]), float(refusal[1])),
            memory_counters=memory,
        )
    return counters, memory


@final
class _PlcHostArrays(StrictModule):
    """Host banks shared by PLC source preparation and recovery."""

    points: NDArray[np.float64]
    offsets: NDArray[np.int64]
    loops: NDArray[np.int32]
    facets: NDArray[np.int32]
    regions: NDArray[np.int32]
    edges: NDArray[np.int32]

    def layout_aligned(self) -> bool:
        return (
            self.offsets.ndim == 1
            and self.offsets.shape[0] == self.facets.shape[0] + 1
            and self.loops.ndim == 1
            and self.regions.ndim == 2
            and self.regions.shape[1] == 2
            and self.edges.ndim == 2
            and self.edges.shape[1] == 2
            and (not self.offsets.size or bool(self.offsets[-1] == self.loops.shape[0]))
        )


def _plc_host_arrays(
    points: ArrayLike,
    polygon_offsets: ArrayLike,
    polygon_vertices: ArrayLike,
    polygon_facets: ArrayLike,
    facet_regions: ArrayLike,
    segments: ArrayLike | None,
    /,
) -> _PlcHostArrays:
    point_array = _points(points, "points", 3)
    offset_values = _index_array(polygon_offsets, "polygon_offsets")
    if np.any(offset_values < 0) or np.any(offset_values > np.iinfo(np.int64).max):
        raise ValueError(
            f"_meshcore._plc_host_arrays.polygon_offsets must contain nonnegative signed-int64 values; got {_input_summary(polygon_offsets)}."
        )
    offsets = np.ascontiguousarray(offset_values, dtype=np.int64)
    loops = _power_indices(polygon_vertices, "polygon_vertices")
    facets = _power_indices(polygon_facets, "polygon_facets")
    regions = _power_indices(facet_regions, "facet_regions")
    edges = (
        np.zeros((0, 2), dtype=np.int32)
        if segments is None
        else _power_indices(segments, "segments")
    )
    return _PlcHostArrays(point_array, offsets, loops, facets, regions, edges)


def plc_source_constraints(
    points: ArrayLike,
    polygon_offsets: ArrayLike,
    polygon_vertices: ArrayLike,
    polygon_facets: ArrayLike,
    facet_regions: ArrayLike,
    /,
    *,
    segments: ArrayLike | None = None,
    work_limit: int,
    max_scratch_bytes: int | None = None,
    record_native_phase: NativePhaseRecorder | None = None,
) -> PlcSourceConstraints:
    """Prepare PLC edge/polygon strata without constructing a volume.

    Source ordering is identical to recovery's source preparation. Allocation,
    validation, and work refusals retain native evidence; optional phase
    measurements do not become fields of the returned scientific artifact.
    """
    library = load_meshcore()
    prepared = _plc_host_arrays(
        points, polygon_offsets, polygon_vertices, polygon_facets, facet_regions, segments
    )
    point_array = prepared.points
    offsets = prepared.offsets
    loops = prepared.loops
    facets = prepared.facets
    regions = prepared.regions
    edges = prepared.edges
    if offsets.ndim != 1 or facets.ndim != 1 or not prepared.layout_aligned():
        raise ValueError(
            f"_meshcore.plc_source_constraints.points must satisfy the input contract (plc_source_constraints: misaligned PLC arrays.); got {_input_summary(points)}."
        )
    handle = ctypes.c_void_p()
    status = library["phx_mc_plc3d_source_constraints"](
        point_array.shape[0],
        point_array.ctypes.data,
        facets.shape[0],
        offsets.ctypes.data,
        loops.ctypes.data,
        facets.ctypes.data,
        regions.shape[0],
        regions.ctypes.data,
        edges.shape[0],
        edges.ctypes.data,
        _budget(work_limit, "work_limit"),
        _unsigned_budget(max_scratch_bytes, "max_scratch_bytes"),
        int(record_native_phase is not None),
        ctypes.addressof(handle),
    )
    if not handle.value:
        _raise_for_call(status, "plc3d_source_constraints")
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.plc_source_constraints.points must satisfy the input contract (plc3d_source_constraints: missing result); got {_input_summary(points)}.",
        )
    try:
        counters, memory = _plc_result_evidence(
            library, handle, status, record_native_phase
        )
        sizes = np.empty((6,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_plc3d_sizes"](handle, sizes.ctypes.data), "phx_mc_plc3d_sizes"
        )
        count, cells, faces, pieces, table, triangles = sizes.tolist()
        if count != point_array.shape[0] or cells or faces or pieces:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.plc_source_constraints.points must satisfy the input contract (Source-only PLC preparation produced a volume.); got {_input_summary(points)}.",
            )
        source_points = np.empty((count, 3), dtype=np.float64)
        source_edges = np.empty((table, 2), dtype=np.int32)
        source_triangles = np.empty((triangles, 3), dtype=np.int32)
        source_polygons = np.empty((triangles,), dtype=np.int32)
        empty = np.empty((0,), dtype=np.int32)
        protection = np.empty((count,), dtype=np.float64)
        dimensions = np.empty((count,), dtype=np.int8)
        _raise_for_call(
            library["phx_mc_plc3d_export"](
                handle,
                source_points.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                empty.ctypes.data,
                protection.ctypes.data,
                source_edges.ctypes.data,
                dimensions.ctypes.data,
                source_triangles.ctypes.data,
                source_polygons.ctypes.data,
            ),
            "phx_mc_plc3d_export",
        )
        result = PlcSourceConstraints(
            source_points,
            source_edges,
            source_triangles,
            source_polygons,
            counters,
            memory,
        )
    finally:
        library["phx_mc_plc3d_free"](handle)
    return result


def recover_plc_3d(
    points: ArrayLike,
    polygon_offsets: ArrayLike,
    polygon_vertices: ArrayLike,
    polygon_facets: ArrayLike,
    facet_regions: ArrayLike,
    /,
    *,
    segments: ArrayLike | None = None,
    seeds: ConvertibleToArray | None = None,
    seed_regions: ConvertibleToArray | None = None,
    facet_tolerances: ArrayLike | None = None,
    segment_tolerances: ArrayLike | None = None,
    boundary_policy: TetMeshBoundaryPolicy,
    max_vertices: int,
    max_tetrahedra: int,
    work_limit: int,
    max_scratch_bytes: int | None = None,
    record_native_phase: NativePhaseRecorder | None = None,
    admit_host_output: Callable[[int], None] | None = None,
) -> PlcRecovery3D:
    """Validate an oriented PLC and recover it in a constrained tetrahedralization.

    ``facet_regions`` (F, 2) names the region on the positive side of each
    facet's right-hand loop normal and on its negative side (-1 void). A
    ``"fixed"`` boundary admits no boundary Steiner point; ``"conforming"``
    splits PLC edges and facets at exactly representable points on them or,
    under a positive declared ``facet_tolerances`` (F,) / ``segment_tolerances``
    (S,) bound (``None``: zero, only), at ancestry-backed carriers whose
    certified deviation that bound admits. A scientific refusal raises
    :class:`PlcRecoveryFailure` with its evidence; malformed arguments raise
    ``ValueError``.

    ``max_scratch_bytes=None`` selects ordinary untracked native allocation
    (UINT64_MAX); zero is a hard allocation cap. ``memory_counters`` is present
    on success/failure only when an allocator owner supplied the six
    uint64 measurements, not an estimated size or fabricated zero ledger.
    """

    library = load_meshcore()
    prepared = _plc_host_arrays(
        points, polygon_offsets, polygon_vertices, polygon_facets, facet_regions, segments
    )
    point_array = prepared.points
    offsets = prepared.offsets
    loops = prepared.loops
    facets = prepared.facets
    regions = prepared.regions
    edges = prepared.edges
    seed_points = _points(
        np.zeros((0, 3), dtype=np.float64) if seeds is None else seeds, "seeds", 3
    )
    seed_labels = (
        np.zeros((0,), dtype=np.int32)
        if seed_regions is None
        else _power_indices(seed_regions, "seed_regions")
    )
    facet_bounds = (
        None
        if facet_tolerances is None
        else np.ascontiguousarray(facet_tolerances, dtype=np.float64)
    )
    segment_bounds = (
        None
        if segment_tolerances is None
        else np.ascontiguousarray(segment_tolerances, dtype=np.float64)
    )
    for name, bounds, count in (
        ("facet_tolerances", facet_bounds, regions.shape[0]),
        ("segment_tolerances", segment_bounds, edges.shape[0]),
    ):
        if bounds is not None and (
            bounds.shape != (count,)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
        ):
            raise ValueError(
                f"{'_meshcore.' + name} must be finite, nonnegative and one per entity; got {_input_summary(points)}."
            )
    if not prepared.layout_aligned() or seed_labels.shape != (seed_points.shape[0],):
        raise ValueError(
            f"_meshcore.recover_plc_3d.points must satisfy the input contract (recover_plc_3d: misaligned PLC arrays.); got {_input_summary(points)}."
        )
    match boundary_policy:
        case "fixed":
            policy = 0
        case "conforming":
            policy = 1
        case _:
            raise ValueError(
                f"recover_plc_3d.boundary_policy must be fixed or conforming; got {boundary_policy!r}."
            )
    handle = ctypes.c_void_p()
    status = library["phx_mc_plc3d_recover"](
        ctypes.byref(
            _Plc3DRecoverRequest(
                point_count=point_array.shape[0],
                points=point_array.ctypes.data,
                polygon_count=facets.shape[0],
                polygon_offsets=offsets.ctypes.data,
                polygon_vertices=loops.ctypes.data,
                polygon_facets=facets.ctypes.data,
                facet_count=regions.shape[0],
                facet_regions=regions.ctypes.data,
                segment_count=edges.shape[0],
                segments=edges.ctypes.data,
                seed_count=seed_points.shape[0],
                seeds=seed_points.ctypes.data,
                seed_regions=seed_labels.ctypes.data,
                boundary_policy=policy,
                facet_tolerances=None
                if facet_bounds is None
                else facet_bounds.ctypes.data,
                segment_tolerances=None
                if segment_bounds is None
                else segment_bounds.ctypes.data,
                max_vertices=_budget(max_vertices, "max_vertices"),
                max_tetrahedra=_budget(max_tetrahedra, "max_tetrahedra"),
                work_limit=_budget(work_limit, "work_limit"),
                maximum_scratch_bytes=_unsigned_budget(
                    max_scratch_bytes, "max_scratch_bytes"
                ),
                measure_phases=int(record_native_phase is not None),
                result=ctypes.addressof(handle),
            )
        )
    )
    if not handle.value:
        _raise_for_call(status, "plc3d_recover")
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.recover_plc_3d.points must satisfy the input contract (plc3d_recover: missing result); got {_input_summary(points)}.",
        )
    # The native result is released even when a copy fails.
    try:
        counters, memory = _plc_result_evidence(
            library, handle, status, record_native_phase
        )
        sizes = np.zeros((6,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_plc3d_sizes"](handle, sizes.ctypes.data), "phx_mc_plc3d_sizes"
        )
        count, cells, faces, pieces, table, triangles = (int(value) for value in sizes)
        if any(value < 0 for value in (count, cells, faces, pieces, table, triangles)):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "recover_plc_3d.output_sizes must be nonnegative; got a negative native size.",
            )
        if admit_host_output is not None:
            # Exact payload of the sixteen arrays below, plus their declared
            # 256-byte host wrappers and the returned record. The native result
            # remains live here, so admission covers their real overlap.
            f64 = np.dtype(np.float64).itemsize
            i32 = np.dtype(np.int32).itemsize
            i8 = np.dtype(np.int8).itemsize
            payload = (
                count * (7 * f64 + i32 + 2 * i8)
                + cells * 5 * i32
                + faces * 4 * i32
                + pieces * 3 * i32
                + table * 2 * i32
                + triangles * 4 * i32
            )
            # The callback owns the output array wrappers and returned record.
            # Counter-query temporaries belong to the caller's live input phase.
            bound = payload + 17 * 256
            if bound > np.iinfo(np.uint64).max:
                raise MeshcoreError(
                    MeshcoreStatus.CAPACITY_EXCEEDED,
                    "recover_plc_3d.host_output must fit the managed-byte counter; got an overflowing export size.",
                )
            admit_host_output(bound)
        result = PlcRecovery3D(
            np.zeros((count, 3), dtype=np.float64),
            np.zeros((cells, 4), dtype=np.int32),
            np.zeros((cells,), dtype=np.int32),
            np.zeros((faces, 3), dtype=np.int32),
            np.zeros((faces,), dtype=np.int32),
            np.zeros((pieces, 2), dtype=np.int32),
            np.zeros((pieces,), dtype=np.int32),
            np.zeros((count,), dtype=np.float64),
            np.zeros((table, 2), dtype=np.int32),
            np.zeros((count,), dtype=np.int8),
            np.zeros((triangles, 3), dtype=np.int32),
            np.zeros((triangles,), dtype=np.int32),
            np.zeros((count,), dtype=np.int8),
            np.zeros((count,), dtype=np.int32),
            np.zeros((count, 2), dtype=np.float64),
            np.zeros((count,), dtype=np.float64),
            counters,
            memory,
        )
        _raise_for_call(
            library["phx_mc_plc3d_export"](
                handle,
                result.points.ctypes.data,
                result.tetrahedra.ctypes.data,
                result.tetrahedron_regions.ctypes.data,
                result.faces.ctypes.data,
                result.face_sources.ctypes.data,
                result.segments.ctypes.data,
                result.segment_sources.ctypes.data,
                result.protection_radii.ctypes.data,
                result.plc_edges.ctypes.data,
                result.vertex_dimension.ctypes.data,
                result.input_triangles.ctypes.data,
                result.input_polygons.ctypes.data,
            ),
            "phx_mc_plc3d_export",
        )
        _raise_for_call(
            library["phx_mc_plc3d_source_witnesses"](
                handle,
                result.witness_strata.ctypes.data,
                result.witness_entities.ctypes.data,
                result.witness_parameters.ctypes.data,
                result.witness_deviations.ctypes.data,
                None,
            ),
            "phx_mc_plc3d_source_witnesses",
        )
    finally:
        library["phx_mc_plc3d_free"](handle)
    return result


# ------------------------------------------------ constrained tetrahedral mesh


TET_MESH_UNMET_CRITERIA = ("radius_edge", "size", "dihedral", "construction", "validity")
TET_MESH_UNMET_REASONS = (
    "budget",
    "fixed_boundary",
    "protected",
    "nonrepresentable",
    "refused",
    "no_improvement",
)
_TET_MESH_QUALITY_BINS = 18


@final
class TetMeshRun[
    CounterT: TetMeshRefineCounters | TetMeshImproveCounters | TetMeshExudeCounters
](StrictModule):
    """Status and counters of one refinement or improvement run.

    ``status`` is ``OK`` (every criterion met), ``REFINEMENT_LIMIT`` (valid
    mesh; unmet cells listed by :meth:`TetMesh3D.unmet`) or
    ``CAPACITY_EXCEEDED`` (a work, vertex or cell limit stopped the run; the
    mesh is valid). ``counters`` follow :class:`TetMeshRefineCounters`,
    :class:`TetMeshImproveCounters` or :class:`TetMeshExudeCounters`.
    """

    status: MeshcoreStatus = eqx.field(static=True)
    counters: CounterT


@final
class TetMeshQuality(StrictModule):
    """Shape distribution of the finite cells (angles in degrees).

    ``dihedral_histogram`` counts per-cell smallest dihedral angles in 5 degree
    bins over ``[0, 90)``; ``slivers`` counts cells below the requested angle.
    """

    minimum_dihedral: float = eqx.field(static=True)
    maximum_dihedral: float = eqx.field(static=True)
    maximum_radius_edge: float = eqx.field(static=True)
    mean_radius_edge: float = eqx.field(static=True)
    minimum_volume: float = eqx.field(static=True)
    total_volume: float = eqx.field(static=True)
    mean_minimum_dihedral: float = eqx.field(static=True)
    dihedral_histogram: np.ndarray
    slivers: int = eqx.field(static=True)


@final
class TetMeshUnmet(StrictModule):
    """Cells that violate a criterion after the last run and why.

    ``criteria`` index :data:`TET_MESH_UNMET_CRITERIA`, ``reasons``
    :data:`TET_MESH_UNMET_REASONS`; ``values`` are the measured radius-edge
    ratio, circumradius-to-size ratio, smallest dihedral angle (degrees), the
    shared floating orientation filter's positive-determinant lower bound, or
    for ``validity`` the ``det**2 / (floor**2 * prod |e_k|**2) - 1`` margin
    of the relative determinant floor (``e_k`` the edges from the cell's
     first vertex; nonpositive below the floor).
    """

    tetrahedra: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray


@final
class TetMeshArrays(StrictModule):
    """Snapshot of a :class:`TetMesh3D`.

    Tetrahedra are positively oriented, rotated to start at their smallest
    vertex and sorted; faces are oriented outward from the domain (from the
    lower region at interfaces); segments are sorted. ``points`` keep every
    vertex id; removed vertices have ``vertex_dimension`` -1 (else 0 corner,
    1 segment, 2 facet, 3 interior).
    """

    points: np.ndarray
    tetrahedra: np.ndarray
    tetrahedron_regions: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    vertex_dimension: np.ndarray
    vertex_sizes: np.ndarray


@final
class TetMeshInsertionPlan(StrictModule):
    """Immutable inspected final conflict cavity tied to one accepted native state.

    The input source construction record and finite cell tuples own no reconstructed
    mesh/classification. The native owner rechecks their generation and
    final geometry/source reconstruction before its one atomic commit.
    """

    _owner: TetMesh3D = eqx.field(static=True)
    first: int = eqx.field(static=True)
    second: int = eqx.field(static=True)
    split_position: NDArray[np.float64]
    position: NDArray[np.float64]
    source_fraction: float = eqx.field(static=True)
    maximum_cavity_cells: int = eqx.field(static=True)
    insertion_kind: int = eqx.field(static=True)
    accepted_generation: int = eqx.field(static=True)
    removed: NDArray[np.int32]
    proposed: NDArray[np.int32]
    buffer_bytes: int = eqx.field(static=True)
    projected_source: int = eqx.field(static=True, default=-1)
    projected_faces: NDArray[np.int32] | None = None


@final
class TetMeshCurvedSources(StrictModule):
    """Host-declared smooth (curved) face sources of the own source complex.

    ``source_ids`` (K,) are sorted scientific face source labels whose faces
    belong to one smooth source surface, with finite nonnegative
    ``tolerances`` (K,) and 32-byte ``digests`` (K, 32) binding each source to
    the host certificate identity (source, revision, profile and limits).
    ``face_claims`` (F,) give each input face its host-certified whole-facet
    deviation, NaN exactly for faces of undeclared sources. Native only
    compares digests and bounds; claims remain host evidence, never native
    or planar PLC proof.
    """

    source_ids: NDArray[np.int64]
    tolerances: NDArray[np.float64]
    digests: NDArray[np.uint8]
    face_claims: NDArray[np.float64]

    def __post_init__(self) -> None:
        banks = (
            (self.source_ids, np.dtype(np.int64), 1),
            (self.tolerances, np.dtype(np.float64), 1),
            (self.digests, np.dtype(np.uint8), 2),
            (self.face_claims, np.dtype(np.float64), 1),
        )
        for bank, dtype, rank in banks:
            if (
                not isinstance(bank, np.ndarray)
                or bank.dtype != dtype
                or bank.ndim != rank
            ):
                raise TypeError(
                    f"_meshcore.TetMeshCurvedSources.__post_init__.self must satisfy the input contract (Curved source banks require their host dtypes and ranks.); got {_input_summary(self)}."
                )
        count = self.source_ids.shape[0]
        if (
            count == 0
            or self.tolerances.shape != (count,)
            or self.digests.shape != (count, 32)
            or np.any(self.source_ids < 0)
            or np.any(self.source_ids[1:] <= self.source_ids[:-1])
            or not np.all(np.isfinite(self.tolerances))
            or np.any(self.tolerances < 0.0)
        ):
            raise ValueError(
                f"_meshcore.TetMeshCurvedSources.__post_init__.self must declare distinct sorted sources with finite nonnegative tolerances and 32-byte digests; got {_input_summary(self)}."
            )
        for bank, _, _ in banks:
            bank.setflags(write=False)


@final
class TetMeshCurvedCollapsePlan(StrictModule):
    """Validated, unpublished curved-source collapse of ``remove`` onto ``keep``.

    ``faces`` (k, 3) are exactly the new constrained faces, oriented outward
    from the source stratum's first material side (outward from the domain at
    a boundary), on curved source ``source`` in ``accepted_generation``. The
    host must certify every whole facet before :meth:`TetMesh3D.collapse_curved_edge`.
    """

    _owner: TetMesh3D = eqx.field(static=True)
    remove: int = eqx.field(static=True)
    keep: int = eqx.field(static=True)
    source: int = eqx.field(static=True)
    source_row: int = eqx.field(static=True)
    accepted_generation: int = eqx.field(static=True)
    require_improvement: bool = eqx.field(static=True)
    faces: NDArray[np.int32]
    removed: NDArray[np.int32]
    proposed: NDArray[np.int32]


@final
class TetMeshCurvedClaims(StrictModule):
    """Retained delegated curved-source host claims of a native mesh.

    Per claimed source row: ``rows`` (native face rows), scientific
    ``source_ids``, bitwise ``coordinates`` (k, 3, 3), host-certified
    ``deviations`` and the accepted ``generations`` in which the rows entered
    the bank (initial zero; committed rows use the resulting accepted epoch).
    Claims are host evidence; publication recomputes them from the original source.
    """

    rows: NDArray[np.int32]
    source_ids: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    deviations: NDArray[np.float64]
    generations: NDArray[np.int64]


def _free_tet_mesh(library: MeshcoreLibrary, handle: int, /) -> None:
    library["phx_mc_tet_mesh_free"](handle)


def _rows(value: object, name: str, width: int, /) -> np.ndarray:
    array = _power_indices(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{'_meshcore.' + name} must have shape (n, {width}); got {_input_summary(value)}."
        )
    return array


def _labels(value: object, name: str, count: int, /) -> np.ndarray:
    array = _power_indices(value, name)
    if array.shape != (count,):
        raise ValueError(
            f"{'_meshcore.' + name} must have shape ({count},); got {_input_summary(value)}."
        )
    return array


def _scientific_source_labels(
    value: object, name: str, count: int, /
) -> NDArray[np.int64]:
    labels = _index_array(value, name)
    if (
        labels.shape != (count,)
        or np.any(labels < 0)
        or np.any(labels > np.iinfo(np.int64).max)
    ):
        raise ValueError(
            f"{'_meshcore.' + name} must contain nonnegative signed-int64 scientific labels with shape ({count},); got {_input_summary(value)}."
        )
    return np.ascontiguousarray(labels, dtype=np.int64)


def _source_group_rows(
    labels: NDArray[np.int64],
    scientific_ids: NDArray[np.int64],
    name: str,
    /,
) -> NDArray[np.int32]:
    """Lower IDs only through their explicit immutable scientific bank."""
    positions = np.searchsorted(scientific_ids, labels)
    if labels.size and (
        not scientific_ids.size
        or np.any(positions >= scientific_ids.size)
        or not np.array_equal(
            scientific_ids[np.minimum(positions, scientific_ids.size - 1)], labels
        )
    ):
        raise ValueError(
            f"{'_meshcore.' + name} must satisfy the input contract ({name} names a source absent from its declared scientific ID bank.); got {_input_summary(name)}."
        )
    if scientific_ids.size > _MAX_POINTS:
        raise ValueError(
            f"_meshcore._source_group_rows.labels must satisfy the input contract (Native source group row count exceeds the signed-int32 indexing capacity.); got {_input_summary(labels)}."
        )
    return np.ascontiguousarray(positions, dtype=np.int32)


def _budget(value: int | None, name: str, /) -> int:
    if value is None:
        return 2**63 - 1
    return bounded_integer(value, name, lower=0, upper=2**63 - 1)


def _unsigned_budget(value: int | None, name: str, /) -> int:
    if value is None:
        return 2**64 - 1
    if type(value) is int and 0 <= value <= _MAX_COUNTER_VALUE:
        return value
    return bounded_integer(value, name, lower=0, upper=_MAX_COUNTER_VALUE)


def _node_id(value: int, name: str, /) -> int:
    return bounded_integer(value, name, lower=0, upper=2**31 - 1)


def _boolean_flag(value: bool, name: str, /) -> int:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{'_meshcore.' + name} must be boolean; got {_input_summary(value)}."
        )
    return int(value)


def _run_status(status: int, operation: str, /) -> MeshcoreStatus:
    code = MeshcoreStatus(status)
    match code:
        case (
            MeshcoreStatus.OK
            | MeshcoreStatus.REFINEMENT_LIMIT
            | MeshcoreStatus.CAPACITY_EXCEEDED
        ):
            return code
        case _:
            _raise_for_call(status, operation)
            raise MeshcoreError(
                code,
                f"_meshcore._run_status.status must satisfy the input contract ({operation}: unexpected status); got {_input_summary(status)}.",
            )


@final
class TetMeshEdgeConstruction(StrictModule):
    """Nonmutating native edge proposal and its input source-row construction record."""

    position: NDArray[np.float64]
    parameter: float = eqx.field(static=True)
    witness_stratum: int = eqx.field(static=True)
    witness_entity: int = eqx.field(static=True)
    witness_parameters: NDArray[np.float64]
    witness_deviation: float = eqx.field(static=True)


@final
class TetMeshSourceComplex(StrictModule):
    """Declared source complex with an independent immutable source-point bank.

    ``faces`` (R, 3) and ``segments`` (Q, 2) index ``points``, with
    native row selectors ``face_group_rows``/``segment_group_rows`` into the
      int64 scientific banks ``face_ids``/``segment_ids``.
    Deviation bounds ``face_tolerances``/``segment_tolerances`` are nonnegative
    (zero: ); a group row is never a scientific identity.
    Per initial point, ``witness_strata`` (0 none, 1 segment row, 2 triangle
    row) with ``witness_entities`` and ``witness_parameters`` (N, 2): a line
    parameter from the row's first endpoint, or the weights of the triangle
    row's second and third corners. The native owner recertifies every
    construction record deviation and every face/segment ancestry; nothing is trusted.
    """

    points: NDArray[np.float64]
    faces: NDArray[np.int32]
    face_group_rows: NDArray[np.int32]
    face_tolerances: NDArray[np.float64]
    segments: NDArray[np.int32]
    segment_group_rows: NDArray[np.int32]
    segment_tolerances: NDArray[np.float64]
    witness_strata: NDArray[np.int8]
    witness_entities: NDArray[np.int32]
    witness_parameters: NDArray[np.float64]
    face_ids: NDArray[np.int64]
    segment_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        banks = (
            (self.points, np.dtype(np.float64), 2),
            (self.faces, np.dtype(np.int32), 2),
            (self.face_group_rows, np.dtype(np.int32), 1),
            (self.face_tolerances, np.dtype(np.float64), 1),
            (self.segments, np.dtype(np.int32), 2),
            (self.segment_group_rows, np.dtype(np.int32), 1),
            (self.segment_tolerances, np.dtype(np.float64), 1),
            (self.witness_strata, np.dtype(np.int8), 1),
            (self.witness_entities, np.dtype(np.int32), 1),
            (self.witness_parameters, np.dtype(np.float64), 2),
            (self.face_ids, np.dtype(np.int64), 1),
            (self.segment_ids, np.dtype(np.int64), 1),
        )
        for bank, dtype, rank in banks:
            if not isinstance(bank, np.ndarray) or bank.dtype != dtype:
                raise TypeError(
                    f"_meshcore.TetMeshSourceComplex.__post_init__.self must satisfy the input contract (Declared native PLC source banks require their host dtypes.); got {_input_summary(self)}."
                )
            if bank.ndim != rank:
                raise ValueError(
                    f"_meshcore.TetMeshSourceComplex.__post_init__.self must satisfy the input contract (Declared native PLC source banks have invalid ranks.); got {_input_summary(self)}."
                )
        for groups, identifiers in (
            (self.face_group_rows, self.face_ids),
            (self.segment_group_rows, self.segment_ids),
        ):
            if np.any(identifiers < 0) or np.any(identifiers[1:] <= identifiers[:-1]):
                raise ValueError(
                    f"_meshcore.TetMeshSourceComplex.__post_init__.self must be nonnegative, distinct and canonically ordered; got {_input_summary(self)}."
                )
            if np.any(groups < 0) or np.any(groups >= identifiers.size):
                raise ValueError(
                    f"_meshcore.TetMeshSourceComplex.__post_init__.self must satisfy the input contract (Native source group rows index undeclared scientific IDs.); got {_input_summary(self)}."
                )
        _source_complex(self, self.witness_strata.shape[0])
        for bank, _, _ in banks:
            bank.setflags(write=False)


@final
class TetMeshSourceEvidence(StrictModule):
    """Source fidelity of a native tetrahedral mesh.

    ``requested_bound`` is the largest declared row bound and
    ``achieved_bound`` the largest certified deviation of a live vertex (zero
    exactly when every vertex lies on its source). ``face_ancestor_offsets``
    and ``face_ancestor_rows`` retain sorted positive-area source primitive
    contributors as CSR, proving nonoverlapping facet coverage without gaps.
    ``segment_ancestors`` names one source row per segment (-1 none). Per
    vertex the primitive construction row, parameters and certified deviation
    remain independent of facet contributors. ``refusals`` are the bounded carriers
    refused: represented edges (K, 2), stratum, source row, certified
    deviation and declared bound.
    """

    requested_bound: float = eqx.field(static=True)
    achieved_bound: float = eqx.field(static=True)
    face_ancestor_offsets: NDArray[np.int64]
    face_ancestor_rows: NDArray[np.int32]
    segment_ancestors: NDArray[np.int32]
    witness_strata: NDArray[np.int8]
    witness_entities: NDArray[np.int32]
    witness_parameters: NDArray[np.float64]
    witness_deviations: NDArray[np.float64]
    refused_edges: NDArray[np.int32]
    refused_strata: NDArray[np.int8]
    refused_entities: NDArray[np.int32]
    refused_deviations: NDArray[np.float64]
    refused_bounds: NDArray[np.float64]
    face_ids: NDArray[np.int64]
    segment_ids: NDArray[np.int64]


def _source_complex(source: TetMeshSourceComplex, count: int, /) -> NativeSourceComplex:
    points = _points(source.points, "source points", 3)
    faces = _rows(source.faces, "source faces", 3)
    face_ids = _labels(source.face_group_rows, "source face_group_rows", faces.shape[0])
    face_bounds = _real_array(source.face_tolerances, "source face_tolerances")
    segments = _rows(source.segments, "source segments", 2)
    segment_ids = _labels(
        source.segment_group_rows, "source segment_group_rows", segments.shape[0]
    )
    segment_bounds = _real_array(source.segment_tolerances, "source segment_tolerances")
    strata = np.ascontiguousarray(source.witness_strata, dtype=np.int8)
    entities = np.ascontiguousarray(source.witness_entities, dtype=np.int32)
    parameters = _real_array(source.witness_parameters, "witness_parameters")
    if (
        face_bounds.shape != (faces.shape[0],)
        or segment_bounds.shape != (segments.shape[0],)
        or strata.shape != (count,)
        or entities.shape != (count,)
        or parameters.shape != (count, 2)
    ):
        raise ValueError(
            f"_meshcore._source_complex.source must satisfy the input contract (The declared source complex is misaligned with the carrier.); got {_input_summary(source)}."
        )
    return NativeSourceComplex(
        faces=faces,
        face_group_rows=face_ids,
        face_tolerances=face_bounds,
        segments=segments,
        segment_group_rows=segment_ids,
        segment_tolerances=segment_bounds,
        witness_strata=strata,
        witness_entities=entities,
        witness_parameters=parameters,
        points=points,
    )


@final
class TetMesh3D:
    """Owned native tetrahedral mesh of a constrained domain.

    Built from positively oriented domain ``tetrahedra`` (T, 4) with
    ``tetrahedron_regions`` (T,) >= 0, constrained ``faces`` (F, 3) with
    ``face_sources`` (F,) >= 0 covering every domain-boundary face and region
    interface, and protected ``segments`` (S, 2) with ``segment_sources``
    (S,) >= 0. ``protection_radii`` (N,) are protecting-ball radii (0
    unprotected). Without ``source`` the initial faces and segments are their
    own source complex; a :class:`TetMeshSourceComplex` declares the
    source rows, their deviation bounds and the initial witnesses. A
    face edge that is not a segment must be shared by exactly two faces of one
    source, coplanar unless a vertex carries a bounded construction record or
    both faces belong to one ``curved`` source (:class:`TetMeshCurvedSources`,
    own source complex only). A ``"fixed"``
    boundary never splits constrained faces or segments; ``"conforming"``
    splits them at points exactly on them or, when none is representable, at
    ancestry-backed carriers whose certified deviation the source row's bound
    admits; children keep source, region and orientation. Every change is one
    validated native transaction; refusals leave the mesh unchanged. An
    invalid domain raises ``ValueError``. The native state is released by
    :meth:`close` or on collection.
    """

    __slots__ = (
        "__weakref__",
        "_finalizer",
        "_handle",
        "_library",
        "_face_source_ids",
        "_segment_source_ids",
    )

    @checked
    def __init__(
        self,
        points: ArrayLike,
        tetrahedra: ArrayLike,
        tetrahedron_regions: ArrayLike,
        faces: ArrayLike,
        face_sources: ArrayLike,
        segments: ArrayLike,
        segment_sources: ArrayLike,
        /,
        *,
        boundary_policy: TetMeshBoundaryPolicy,
        protection_radii: ArrayLike | None = None,
        max_vertices: int | None = None,
        max_tetrahedra: int | None = None,
        max_scratch_bytes: int | None = None,
        source: TetMeshSourceComplex | None = None,
        curved: TetMeshCurvedSources | None = None,
    ) -> None:
        library = load_meshcore()
        point_array = _points(points, "points", 3)
        count = point_array.shape[0]
        cells = _rows(tetrahedra, "tetrahedra", 4)
        regions = _labels(tetrahedron_regions, "tetrahedron_regions", cells.shape[0])
        face_rows = _rows(faces, "faces", 3)
        sources = _scientific_source_labels(
            face_sources, "face_sources", face_rows.shape[0]
        )
        segment_rows = _rows(segments, "segments", 2)
        segment_labels = _scientific_source_labels(
            segment_sources, "segment_sources", segment_rows.shape[0]
        )
        radii = None
        if protection_radii is not None:
            radii = _real_array(protection_radii, "protection_radii")
            if radii.shape != (count,):
                raise ValueError(
                    f"_meshcore.TetMesh3D.__init__.protection_radii must have shape ({count},); got {_input_summary(protection_radii)}."
                )
        match boundary_policy:
            case "fixed":
                policy = 0
            case "conforming":
                policy = 1
            case _:
                raise ValueError(
                    f"TetMesh3D.boundary_policy must be fixed or conforming; got {boundary_policy!r}."
                )
        vertices = _limit(max_vertices, _MAX_POINTS, "max_vertices")
        if vertices < count or vertices > _MAX_POINTS:
            raise ValueError(
                f"TetMesh3D.max_vertices must lie in [{count}, {_MAX_POINTS}]; got {vertices}."
            )
        tetrahedra_limit = _limit(max_tetrahedra, 2**30 - 1, "max_tetrahedra")
        declared = None if source is None else _source_complex(source, count)
        face_ids = np.unique(sources) if source is None else source.face_ids
        segment_ids = np.unique(segment_labels) if source is None else source.segment_ids
        face_group_rows = _source_group_rows(sources, face_ids, "face_sources")
        segment_group_rows = _source_group_rows(
            segment_labels, segment_ids, "segment_sources"
        )
        curved_rows, curved_tolerances, curved_digests, face_claims = (
            None,
            None,
            None,
            None,
        )
        if curved is not None:
            if source is not None:
                raise ValueError(
                    f"_meshcore.TetMesh3D.__init__.curved must use the own source complex; got a declared source with {_input_summary(curved)}."
                )
            if curved.face_claims.shape != (face_rows.shape[0],):
                raise ValueError(
                    f"_meshcore.TetMesh3D.__init__.curved.face_claims must have shape ({face_rows.shape[0]},); got {_input_summary(curved.face_claims)}."
                )
            curved_rows = _source_group_rows(
                curved.source_ids, face_ids, "curved sources"
            )
            curved_tolerances = np.ascontiguousarray(curved.tolerances)
            curved_digests = np.ascontiguousarray(curved.digests)
            face_claims = np.ascontiguousarray(curved.face_claims)
        handle = ctypes.c_void_p()
        status = library["phx_mc_tet_mesh_create"](
            ctypes.byref(
                _TetMeshCreateRequest(
                    point_count=count,
                    points=point_array.ctypes.data,
                    tet_count=cells.shape[0],
                    tets=cells.ctypes.data,
                    tet_regions=regions.ctypes.data,
                    face_count=face_rows.shape[0],
                    faces=face_rows.ctypes.data,
                    face_sources=face_group_rows.ctypes.data,
                    segment_count=segment_rows.shape[0],
                    segments=segment_rows.ctypes.data,
                    segment_sources=segment_group_rows.ctypes.data,
                    protection_radii=_pointer(radii),
                    source_point_count=0
                    if declared is None
                    else declared.points.shape[0],
                    source_points=None
                    if declared is None
                    else declared.points.ctypes.data,
                    source_face_count=0 if declared is None else declared.faces.shape[0],
                    source_faces=None if declared is None else declared.faces.ctypes.data,
                    source_face_group_rows=None
                    if declared is None
                    else declared.face_group_rows.ctypes.data,
                    source_face_tolerances=None
                    if declared is None
                    else declared.face_tolerances.ctypes.data,
                    source_segment_count=0
                    if declared is None
                    else declared.segments.shape[0],
                    source_segments=None
                    if declared is None
                    else declared.segments.ctypes.data,
                    source_segment_group_rows=None
                    if declared is None
                    else declared.segment_group_rows.ctypes.data,
                    source_segment_tolerances=None
                    if declared is None
                    else declared.segment_tolerances.ctypes.data,
                    witness_strata=None
                    if declared is None
                    else declared.witness_strata.ctypes.data,
                    witness_entities=None
                    if declared is None
                    else declared.witness_entities.ctypes.data,
                    witness_parameters=None
                    if declared is None
                    else declared.witness_parameters.ctypes.data,
                    curved_source_count=0 if curved_rows is None else curved_rows.size,
                    curved_sources=_pointer(curved_rows),
                    curved_tolerances=_pointer(curved_tolerances),
                    curved_digests=_pointer(curved_digests),
                    face_claims=_pointer(face_claims),
                    boundary_policy=policy,
                    max_vertices=vertices,
                    max_tetrahedra=tetrahedra_limit,
                    max_scratch_bytes=_budget(max_scratch_bytes, "max_scratch_bytes"),
                    mesh=ctypes.addressof(handle),
                )
            )
        )
        _raise_for_call(status, "tet_mesh_create")
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.TetMesh3D.__init__.points must satisfy the input contract (tet_mesh_create: missing handle); got {_input_summary(points)}.",
            )
        self._library = library
        self._handle = handle.value
        self._face_source_ids, self._segment_source_ids = face_ids, segment_ids
        face_ids.setflags(write=False)
        segment_ids.setflags(write=False)
        self._finalizer = weakref.finalize(self, _free_tet_mesh, library, handle.value)

    def _live(self) -> int:
        if not self._finalizer.alive:
            raise ValueError(
                f"_meshcore.TetMesh3D._live.self must satisfy the input contract (the tetrahedral mesh is closed.); got {_input_summary(self)}."
            )
        return self._handle

    def close(self) -> None:
        """Release the native state; later calls raise ``ValueError``."""

        self._finalizer()

    def _counts(self) -> np.ndarray:
        counts = np.zeros((6,), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_counts"](self._live(), counts.ctypes.data)
        _raise_for_call(status, "tet_mesh_counts")
        return counts

    def set_work_limit(self, work_limit: int, /) -> None:
        """Allow this many additional native work units from the current total.

        Unbudgeted operations consume this shared remaining allowance. Editing
        operations with their own work budget may only narrow it, then restore
        the input absolute ceiling. :meth:`proposal_shape` additionally caps
        its own query work without replacing or resetting that shared allowance.
        """
        if work_limit is None:
            raise TypeError(
                f"_meshcore.TetMesh3D.set_work_limit.work_limit must be an integer; got {_input_summary(work_limit)}."
            )
        _raise_for_call(
            self._library["phx_mc_tet_mesh_set_work_limit"](
                self._live(), _budget(work_limit, "work_limit")
            ),
            "tet_mesh_set_work_limit",
        )

    def work_units(self) -> int:
        """Return the cumulative native work consumed by this mesh."""
        work = np.zeros((1,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_work_units"](
                self._live(),
                work.ctypes.data,
            ),
            "tet_mesh_work_units",
        )
        return int(work[0])

    def _refine_once(
        self,
        sizes: np.ndarray | None,
        radius_edge_bound: float,
        insertions: int,
        work: int,
        record_native_phase: NativePhaseRecorder | None = None,
    ) -> TetMeshRun[TetMeshRefineCounters]:
        counters = np.zeros((10,), dtype=np.int64)
        if record_native_phase is not None:
            self.measure_execution(True)
        try:
            status = self._library["phx_mc_tet_mesh_refine"](
                self._live(),
                _pointer(sizes),
                radius_edge_bound,
                insertions,
                work,
                counters.ctypes.data,
            )
            if record_native_phase is not None:
                seconds, measured = self.execution_times()
                if not measured[0]:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        f"_meshcore.TetMesh3D._refine_once.sizes must satisfy the input contract (unmeasured refinement invocation); got {_input_summary(sizes)}.",
                    )
                recorded_work = (
                    int(counters[7])
                    if status in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT)
                    else None
                )
                record_native_phase("refinement", float(seconds[0]), recorded_work, 1)
            return TetMeshRun(
                _run_status(status, "tet_mesh_refine"),
                TetMeshRefineCounters.from_slots(counters),
            )
        finally:
            if record_native_phase is not None:
                self.measure_execution(False)

    def refine(
        self,
        *,
        radius_edge_bound: float = 2.0,
        sizes: object | Callable[[np.ndarray], object] | None = None,
        max_insertions: int | None = None,
        work_limit: int | None = None,
        max_rounds: int = 8,
        record_native_phase: NativePhaseRecorder | None = None,
    ) -> TetMeshRun[TetMeshRefineCounters]:
        """Constrained Delaunay refinement to radius-edge and size targets.

        ``sizes`` is ``None`` (keep the current per-vertex targets; new vertices
        interpolate them), a per-vertex array ``(N,) >= 0`` (0: no target), or
        a callable evaluated on the ``(m, 3)`` coordinates of a batch of
        vertices. A callable is sampled on every vertex, then on the vertices
        each round created, for at most ``max_rounds`` native rounds that
        share ``max_insertions`` and ``work_limit``. Cells with circumradius
        above the mean target of their vertices, or radius-edge ratio above
        ``radius_edge_bound``, are refined.
        """

        if not radius_edge_bound > 0.0:
            raise ValueError(
                f"_meshcore.TetMesh3D.refine.radius_edge_bound must be positive; got {_input_summary(radius_edge_bound)}."
            )
        insertions = _budget(max_insertions, "max_insertions")
        work = _budget(work_limit, "work_limit")
        if not callable(sizes):
            targets = None
            if sizes is not None:
                targets = _real_array(sizes, "sizes")
                if targets.shape != (int(self._counts()[0]),):
                    raise ValueError(
                        f"_meshcore.TetMesh3D.refine.sizes must have one value per vertex; got {_input_summary(sizes)}."
                    )
            return self._refine_once(
                targets, radius_edge_bound, insertions, work, record_native_phase
            )
        if max_rounds < 1:
            raise ValueError(
                f"_meshcore.TetMesh3D.refine.max_rounds must be positive; got {_input_summary(max_rounds)}."
            )
        values = np.zeros((0,), dtype=np.float64)
        total = TetMeshRefineCounters()
        run = TetMeshRun(MeshcoreStatus.OK, total)
        for _ in range(max_rounds):
            snapshot = self.arrays()
            fresh = snapshot.points[values.shape[0] :]
            sampled = _real_array(sizes(fresh), "sizes")
            if sampled.shape != (fresh.shape[0],):
                raise ValueError(
                    f"_meshcore.TetMesh3D.refine.radius_edge_bound must return one value per point; got {_input_summary(radius_edge_bound)}."
                )
            values = np.concatenate((values, sampled))
            run = self._refine_once(
                values, radius_edge_bound, insertions, work, record_native_phase
            )
            # Work and insertion counters add up; the unmet count and vertex
            # count describe the last round, the largest cavity the maximum.
            current = run.counters
            total = total.accumulated(current)
            added = (
                current.circumcenter_insertions
                + current.subfacet_splits
                + current.subsegment_splits
            )
            insertions -= added
            work -= current.work_units
            if (
                added == 0
                or run.status == MeshcoreStatus.CAPACITY_EXCEEDED
                or insertions <= 0
            ):
                break
        return TetMeshRun(run.status, total)

    def improve(
        self,
        *,
        min_dihedral_degrees: float = 10.0,
        minimum_relative_determinant: float = 0.0,
        max_passes: int = 8,
        work_limit: int | None = None,
    ) -> TetMeshRun[TetMeshImproveCounters]:
        """Remove slivers by face/edge/multiface reconnection, source-preserving
        relocation and unprotected interior-vertex removal.

        Constrained faces, segments and regions never change; each operation
        is committed only when it raises the local smallest dihedral angle.
        ``minimum_relative_determinant`` is the validity policy's
        relative determinant floor: a cell whose determinant does not
        exceed it times the product of its edge lengths is repaired
        under an construction objective, or reported with the
        ``validity`` criterion. ``0`` requires only positive orientation.
        """

        if not 0.0 <= min_dihedral_degrees < 70.5:
            raise ValueError(
                f"_meshcore.TetMesh3D.improve.min_dihedral_degrees must lie in [0, 70.5); got {_input_summary(min_dihedral_degrees)}."
            )
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError(
                f"TetMesh3D.minimum_relative_determinant must lie in [0, 1); got {minimum_relative_determinant}."
            )
        if isinstance(max_passes, bool) or max_passes < 0:
            raise ValueError(
                f"_meshcore.TetMesh3D.improve.max_passes must be a nonnegative integer; got {_input_summary(max_passes)}."
            )
        counters = np.zeros((10,), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_improve"](
            self._live(),
            min_dihedral_degrees,
            minimum_relative_determinant,
            max_passes,
            _budget(work_limit, "work_limit"),
            counters.ctypes.data,
        )
        return TetMeshRun(
            _run_status(status, "tet_mesh_improve"),
            TetMeshImproveCounters.from_slots(counters),
        )

    def _applied(self, status: int, applied: np.ndarray, operation: str, /) -> bool:
        _raise_for_call(status, operation)
        return bool(applied[0])

    def flip_face(self, face: object, /, *, require_improvement: bool = False) -> bool:
        """2-3 flip of the unconstrained interior face ``(3,)``; whether applied."""

        vertices = _power_indices(face, "face")
        if vertices.shape != (3,):
            raise ValueError(
                f"_meshcore.TetMesh3D.flip_face.face must have shape (3,); got {_input_summary(face)}."
            )
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_flip_face"](
            self._live(),
            vertices.ctypes.data,
            _boolean_flag(require_improvement, "require_improvement"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_flip_face")

    def remove_edge(
        self, first: int, second: int, /, *, require_improvement: bool = False
    ) -> bool:
        """Remove an unconstrained interior edge by optimal ring retriangulation."""

        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_remove_edge"](
            self._live(),
            _node_id(first, "first"),
            _node_id(second, "second"),
            _boolean_flag(require_improvement, "require_improvement"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_remove_edge")

    def relocate(
        self, vertex: int, position: object, /, *, require_improvement: bool = False
    ) -> bool:
        """Move an unprotected interior, facet or segment-interior vertex.

        Conforming constrained motion preserves every incident source plane or
        segment interval and recertifies its construction record before the atomic write.
        Automatic improvement does not generate segment-vertex moves.
        """

        target = _real_array(position, "position")
        if target.shape != (3,):
            raise ValueError(
                f"_meshcore.TetMesh3D.relocate.position must have shape (3,); got {_input_summary(position)}."
            )
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_relocate"](
            self._live(),
            _node_id(vertex, "vertex"),
            target.ctypes.data,
            _boolean_flag(require_improvement, "require_improvement"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_relocate")

    def relocate_vertices(
        self,
        vertices: ArrayLike,
        positions: ArrayLike,
        /,
        *,
        radius_edge_bound: float = 0.0,
        minimum_dihedral_degrees: float = 0.0,
        work_limit: int | None = None,
    ) -> bool:
        """Atomically move a source-stratum cavity union, or retain every old row."""
        rows = _power_indices(vertices, "vertices")
        targets = _real_array(positions, "positions")
        if rows.ndim != 1 or targets.shape != (rows.size, 3):
            raise ValueError(
                f"_meshcore.TetMesh3D.relocate_vertices.vertices must have shapes (n,) and (n, 3); got {_input_summary(vertices)}."
            )
        if np.any(np.diff(rows) <= 0):
            raise ValueError(
                f"_meshcore.TetMesh3D.relocate_vertices.vertices must be sorted and unique; got {_input_summary(vertices)}."
            )
        if not np.isfinite(radius_edge_bound) or radius_edge_bound < 0.0:
            raise ValueError(
                f"_meshcore.TetMesh3D.relocate_vertices.radius_edge_bound must be finite and nonnegative; got {_input_summary(radius_edge_bound)}."
            )
        if (
            not np.isfinite(minimum_dihedral_degrees)
            or not 0.0 <= minimum_dihedral_degrees < 180.0
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.relocate_vertices.minimum_dihedral_degrees must be finite and in [0, 180); got {_input_summary(minimum_dihedral_degrees)}."
            )
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_relocate_vertices"](
            self._live(),
            rows.size,
            rows.ctypes.data,
            targets.ctypes.data,
            radius_edge_bound,
            minimum_dihedral_degrees,
            _budget(work_limit, "work_limit"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_relocate_vertices")

    def remove_vertex(self, vertex: int, /, *, require_improvement: bool = False) -> bool:
        """Remove an interior vertex by its best valid collapse."""

        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_remove_vertex"](
            self._live(),
            _node_id(vertex, "vertex"),
            _boolean_flag(require_improvement, "require_improvement"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_remove_vertex")

    def set_memory_limit(self, max_scratch_bytes: int, /) -> None:
        """Change the native allocation cap; refusal preserves the old cap."""
        _raise_for_call(
            self._library["phx_mc_tet_mesh_set_memory_limit"](
                self._live(), _budget(max_scratch_bytes, "max_scratch_bytes")
            ),
            "tet_mesh_set_memory_limit",
        )

    def memory_counters(self) -> NativeExecutionMemoryCounters:
        """Limit/live/peak/requested bytes and allocation/refusal counts."""
        values = np.empty((6,), dtype=np.uint64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_memory_evidence"](
                self._live(), values.ctypes.data
            ),
            "tet_mesh_memory_evidence",
        )
        return NativeExecutionMemoryCounters.from_slots(values)

    def split_edge(
        self,
        first: int,
        second: int,
        position: object,
        /,
        *,
        target_size: float = 0.0,
        work_limit: int | None = None,
        source_fraction: float | None = None,
    ) -> int | None:
        """Atomically split an edge or its bounded source-insertion cavity.

        ``source_fraction`` carries the parameter returned by
        :meth:`edge_split_point`. If an incident construction is absent,
        it permits only the identical recertified source-row RNE carrier under
        the input row's bound, never a snapped point or a changed stratum.
        """
        fraction = 0.0 if source_fraction is None else float(source_fraction)
        if source_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                f"TetMesh3D.source_fraction must be finite and strictly between zero and one; got {source_fraction}."
            )
        point = _real_array(position, "position")
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(
                f"_meshcore.TetMesh3D.split_edge.position must be a finite three-coordinate point; got {_input_summary(position)}."
            )
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError(
                f"TetMesh3D.target_size must be finite and nonnegative; got {target_size}."
            )
        inserted = np.full((1,), -1, dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_split_edge"](
            self._live(),
            _node_id(first, "first"),
            _node_id(second, "second"),
            point.ctypes.data,
            target_size,
            fraction,
            _budget(work_limit, "work_limit"),
            inserted.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_split_edge")
        return None if inserted[0] < 0 else int(inserted[0])

    def split_edge_relocate(
        self,
        first: int,
        second: int,
        split_position: object,
        final_position: object,
        /,
        *,
        target_size: float = 0.0,
        work_limit: int | None = None,
    ) -> int | None:
        """Atomically split an edge at the final new-Steiner position.

        The split construction record must be exactly collinear and strictly inside the old
        edge. The final point preserves the scientific curve interval or
         oriented source-plane/material patch union, and the closed star
        shell. Only final positive cells are published; shape policy belongs to
        the caller. Local refusal returns None without an intermediate vertex.
        """
        witness = _real_array(split_position, "split_position")
        if witness.shape != (3,) or not np.all(np.isfinite(witness)):
            raise ValueError(
                f"TetMesh3D.split_position must be a finite three-coordinate point; got {_input_summary(witness)}."
            )
        target = _real_array(final_position, "final_position")
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"TetMesh3D.final_position must be a finite three-coordinate point; got {_input_summary(target)}."
            )
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError(
                f"TetMesh3D.target_size must be finite and nonnegative; got {target_size}."
            )
        inserted = np.full((1,), -1, dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_split_edge_relocate"](
            self._live(),
            _node_id(first, "first"),
            _node_id(second, "second"),
            witness.ctypes.data,
            target.ctypes.data,
            target_size,
            _budget(work_limit, "work_limit"),
            inserted.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_split_edge_relocate")
        return None if inserted[0] < 0 else int(inserted[0])

    def inspect_projected_edge_insertion(
        self,
        first: int,
        second: int,
        split_position: object,
        final_position: object,
        source: int,
        /,
        *,
        maximum_cavity_cells: int,
        source_fraction: float,
        work_limit: int | None = None,
    ) -> TetMeshInsertionPlan | None:
        """Inspect one curved-source physical chart over an exact material cavity.

        The complete source-coordinate cavity is validated before its pending
        point is projected. Every new constrained face then requires an explicit
        whole-facet host claim at commit; no source tolerance is substituted for
        exact material coverage.
        """
        return self.inspect_edge_insertion(
            first, second, split_position, final_position,
            maximum_cavity_cells=maximum_cavity_cells,
            source_fraction=source_fraction, work_limit=work_limit,
            projected_source=source,
        )

    def inspect_edge_insertion(
        self,
        first: int,
        second: int,
        split_position: object,
        final_position: object,
        /,
        *,
        maximum_cavity_cells: int,
        work_limit: int | None = None,
        source_fraction: float | None = None,
        projected_source: int | None = None,
    ) -> TetMeshInsertionPlan | None:
        """Inspect the expanded final insertion cavity without publishing.

        Source/material/ghost reconstruction belongs to the existing native
        constrained preparation. Finite removed+proposed rows share the given
        input cavity allowance. Refusals leave accepted geometry unchanged.
        ``source_fraction`` authenticates the input bounded source construction record
        exactly as in :meth:`split_edge`; inspection and commit retain the same
        fraction, RNE carrier and entire native conflict cavity.
        """
        witness = _real_array(split_position, "split_position")
        position = _real_array(final_position, "final_position")
        if witness.shape != (3,) or not np.all(np.isfinite(witness)):
            raise ValueError(
                f"TetMesh3D.split_position must be a finite three-coordinate point; got {_input_summary(witness)}."
            )
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                f"TetMesh3D.final_position must be a finite three-coordinate point; got {_input_summary(position)}."
            )
        fraction = 0.0 if source_fraction is None else float(source_fraction)
        if source_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                f"TetMesh3D.source_fraction must be finite and strictly between zero and one; got {source_fraction}."
            )
        source_row = -1
        if projected_source is not None:
            if source_fraction is None:
                raise ValueError("Projected insertion requires an explicit source fraction.")
            source_row = int(_source_group_rows(
                np.asarray([projected_source], dtype=np.int64),
                self._face_source_ids, "projected source",
            )[0])
        maximum = _budget(maximum_cavity_cells, "maximum_cavity_cells")
        if maximum > np.iinfo(np.intp).max // ((8 if source_row < 0 else 20) * np.dtype(np.int32).itemsize):
            raise ValueError(
                f"TetMesh3D.maximum_cavity_cells must be addressable; got {maximum}."
            )
        budget = current_native_execution_scope()
        if budget is None:
            removed = np.empty((maximum, 4), dtype=np.int32)
            proposed = np.empty((maximum, 4), dtype=np.int32)
        else:
            removed = budget.allocate_host_array((maximum, 4), np.int32)
            proposed = budget.allocate_host_array((maximum, 4), np.int32)
        faces = None
        counts = np.zeros((4 if source_row < 0 else 5,), dtype=np.int64)
        arguments = (
            self._live(), _node_id(first, "first"), _node_id(second, "second"),
            witness.ctypes.data, position.ctypes.data, fraction,
        )
        if source_row < 0:
            status = self._library["phx_mc_tet_mesh_inspect_edge_insertion"](
                *arguments, maximum, _budget(work_limit, "work_limit"),
                removed.ctypes.data, proposed.ctypes.data, counts.ctypes.data,
            )
        else:
            faces = (
                np.empty((4 * maximum, 3), dtype=np.int32)
                if budget is None
                else budget.allocate_host_array((4 * maximum, 3), np.int32)
            )
            status = self._library["phx_mc_tet_mesh_inspect_projected_edge_insertion"](
                *arguments, source_row, maximum, _budget(work_limit, "work_limit"),
                removed.ctypes.data, proposed.ctypes.data, faces.ctypes.data, counts.ctypes.data,
            )
        _raise_for_call(status, "tet_mesh_inspect_edge_insertion")
        old_count, new_count, kind, generation = (int(value) for value in counts[:4])
        if old_count == 0 or new_count == 0:
            return None
        if old_count < 0 or new_count < 0 or old_count + new_count > maximum:
            raise RuntimeError(
                f"_meshcore.TetMesh3D.inspect_edge_insertion.first must satisfy the input contract (Native insertion inspection exceeded its admitted cavity extent.); got {_input_summary(first)}."
            )
        projected_faces = None
        if faces is not None:
            face_count = int(counts[4])
            if not 0 < face_count <= faces.shape[0]:
                raise RuntimeError("Native projected insertion exceeded its face extent.")
            # Match arrays() and curved-collapse inspection's public outward order.
            faces[:face_count, [1, 2]] = faces[:face_count, [2, 1]]
            faces.setflags(write=False)
            projected_faces = faces[:face_count]
        witness, position = witness.copy(), position.copy()
        for values in (witness, position, removed, proposed):
            values.setflags(write=False)
        return TetMeshInsertionPlan(
            self,
            int(first),
            int(second),
            witness,
            position,
            fraction,
            maximum,
            kind,
            generation,
            removed[:old_count],
            proposed[:new_count],
            int(
                removed.nbytes
                + proposed.nbytes
                + counts.nbytes
                + witness.nbytes
                + position.nbytes
                + (0 if faces is None else faces.nbytes)
            ),
            source_row,
            projected_faces,
        )

    @checked
    def commit_edge_insertion(
        self,
        plan: TetMeshInsertionPlan,
        /,
        *,
        target_size: float = 0.0,
        work_limit: int | None = None,
        claim_digest: bytes | None = None,
        face_coordinates: object | None = None,
        face_deviations: object | None = None,
    ) -> int | None:
        """Reprepare this final source construction record/cavity and commit once.

        A foreign owner, stale accepted-state generation or changed inspected
        tuples never becomes a native reconstruction or a partial publication.
        """
        if plan._owner is not self:
            raise ValueError(
                f"_meshcore.TetMesh3D.commit_edge_insertion.plan must belong to this native owner; got {_input_summary(plan)}."
            )
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError(
                f"TetMesh3D.target_size must be finite and nonnegative; got {target_size}."
            )
        removed = _rows(plan.removed, "plan.removed", 4)
        proposed = _rows(plan.proposed, "plan.proposed", 4)
        witness = _real_array(plan.split_position, "plan.split_position")
        position = _real_array(plan.position, "plan.position")
        if witness.shape != (3,) or position.shape != (3,):
            raise ValueError("Insertion plan positions must each have three coordinates.")
        if not np.all(np.isfinite(witness)) or not np.all(np.isfinite(position)):
            raise ValueError("Insertion plan positions must be finite.")
        if np.any(removed < 0) or np.any(proposed < 0):
            raise ValueError("Insertion plan finite tetrahedra require nonnegative vertex IDs.")
        maximum = _budget(plan.maximum_cavity_cells, "plan.maximum_cavity_cells")
        if removed.shape[0] == 0 or proposed.shape[0] == 0 or removed.shape[0] + proposed.shape[0] > maximum:
            raise ValueError("Insertion plan rows must fit their nonempty cavity allowance.")
        if not np.isfinite(plan.source_fraction) or not 0.0 <= plan.source_fraction < 1.0:
            raise ValueError("Insertion plan source fraction is outside its admitted range.")
        generation = _budget(plan.accepted_generation, "plan.accepted_generation")
        source_row = bounded_integer(
            plan.projected_source, "plan.projected_source", lower=-1,
            upper=max(-1, self._face_source_ids.size - 1),
        )
        binding = corners = deviations = faces = None
        if plan.projected_source >= 0:
            if plan.projected_faces is None:
                raise ValueError("Projected insertion plan requires its complete face bank.")
            if not isinstance(claim_digest, bytes) or len(claim_digest) != 32:
                raise ValueError("Projected insertion claim_digest must contain 32 bytes.")
            corners = np.ascontiguousarray(_real_array(face_coordinates, "face_coordinates"))
            deviations = np.ascontiguousarray(_real_array(face_deviations, "face_deviations"))
            faces = _rows(plan.projected_faces, "plan.projected_faces", 3)
            count = faces.shape[0]
            if count == 0 or np.any(faces < 0) or corners.shape != (count, 3, 3) or deviations.shape != (count,):
                raise ValueError("Projected insertion claims must align with all inspected faces.")
            if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(deviations)) or np.any(deviations < 0):
                raise ValueError("Projected insertion coordinates and deviations must be finite and deviations nonnegative.")
            binding = np.frombuffer(claim_digest, dtype=np.uint8)
        elif claim_digest is not None or face_coordinates is not None or face_deviations is not None:
            raise ValueError("Ordinary insertion does not accept projected face claims.")
        inserted = np.full((1,), -1, dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_commit_edge_insertion"](
            ctypes.byref(
                _TetMeshCommitEdgeInsertionRequest(
                    mesh=self._live(),
                    first=_node_id(plan.first, "plan.first"),
                    second=_node_id(plan.second, "plan.second"),
                    split_position=witness.ctypes.data,
                    final_position=position.ctypes.data,
                    source_fraction=plan.source_fraction,
                    maximum_cavity_cells=maximum,
                    insertion_kind=bounded_integer(plan.insertion_kind, "plan.insertion_kind", lower=0, upper=2),
                    accepted_generation=generation,
                    removed_count=removed.shape[0],
                    removed_tetrahedra=removed.ctypes.data,
                    proposed_count=proposed.shape[0],
                    proposed_tetrahedra=proposed.ctypes.data,
                    target_size=target_size,
                    work_limit=_budget(work_limit, "work_limit"),
                    inserted_vertex=inserted.ctypes.data,
                    projected_source=source_row,
                    claim_count=0 if faces is None else faces.shape[0],
                    claim_digest=None if binding is None else binding.ctypes.data,
                    claim_faces=None if faces is None else faces.ctypes.data,
                    claim_coordinates=None if corners is None else corners.ctypes.data,
                    claim_deviations=None if deviations is None else deviations.ctypes.data,
                )
            )
        )
        _raise_for_call(status, "tet_mesh_commit_edge_insertion")
        return None if inserted[0] < 0 else int(inserted[0])

    def protect_vertices(self, vertices: object, /) -> None:
        """Lock explicitly protected live vertices atomically by their IDs."""
        ids = _power_indices(vertices, "vertices")
        if ids.ndim != 1 or np.any(ids < 0):
            raise ValueError(
                f"_meshcore.TetMesh3D.protect_vertices.vertices must be a nonnegative integer vector; got {_input_summary(vertices)}."
            )
        _raise_for_call(
            self._library["phx_mc_tet_mesh_protect_vertices"](
                self._live(), ids.shape[0], ids.ctypes.data
            ),
            "tet_mesh_protect_vertices",
        )

    def construct_curve_points(
        self,
        vertices: ArrayLike,
        neighbors: ArrayLike,
        desired_positions: ArrayLike,
        /,
        *,
        work_limit: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Construct a batch on source-curve intervals without mutation.

        ``neighbors`` names the two same-source incident segments of each
        smooth curve vertex. Desired physical positions select the dominant
        source coordinate. Returned scalar coordinates are physical, not affine
        fractions. Refused rows have a false mask and NaN outputs.
        """
        rows = _power_indices(vertices, "vertices")
        endpoints = _rows(neighbors, "neighbors", 2)
        desired = _real_array(desired_positions, "desired_positions")
        if (
            rows.ndim != 1
            or endpoints.shape != (rows.size, 2)
            or desired.shape != (rows.size, 3)
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.construct_curve_points.vertices must align as (n,), (n, 2), (n, 3); got {_input_summary(vertices)}."
            )
        points = np.empty((rows.size, 3), dtype=np.float64)
        coordinates = np.empty((rows.size,), dtype=np.float64)
        constructed = np.zeros((rows.size,), dtype=np.int32)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_construct_curve_points"](
                self._live(),
                rows.size,
                rows.ctypes.data,
                endpoints.ctypes.data,
                desired.ctypes.data,
                _budget(work_limit, "work_limit"),
                points.ctypes.data,
                coordinates.ctypes.data,
                constructed.ctypes.data,
            ),
            "tet_mesh_construct_curve_points",
        )
        return points, coordinates, constructed.astype(np.bool_)

    def edge_split_point(
        self,
        first: int,
        second: int,
        /,
        *,
        preferred_fraction: float | None = None,
        work_limit: int | None = None,
    ) -> TetMeshEdgeConstruction | None:
        """Construct an incident point or a bounded source-row RNE carrier.

        None retains the input dyadic construction order. An explicit finite
        fraction strictly between zero and one is attempted exactly, then
        dyadic candidates are ranked by distance if that attempt is refused.
        When no candidate exists, the source construction tries the
        preferred fraction (or one half) under its declared row bound. Pass the
        returned fraction to :meth:`split_edge` as ``source_fraction``;
        no source construction record is inferred from rounded coordinates. None means no
        candidate was admitted, never an approximate source snap.
        """
        fraction = 0.0 if preferred_fraction is None else float(preferred_fraction)
        if preferred_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.edge_split_point.preferred_fraction must be finite and strictly between zero and one; got {_input_summary(preferred_fraction)}."
            )
        point = np.empty((3,), dtype=np.float64)
        parameter = np.empty((1,), dtype=np.float64)
        constructed = np.zeros((1,), dtype=np.int32)
        stratum = np.zeros((1,), dtype=np.int8)
        entity = np.full((1,), -1, dtype=np.int32)
        parameters = np.zeros((2,), dtype=np.float64)
        deviation = np.zeros((1,), dtype=np.float64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_construct_edge_split"](
                self._live(),
                _node_id(first, "first"),
                _node_id(second, "second"),
                fraction,
                _budget(work_limit, "work_limit"),
                point.ctypes.data,
                parameter.ctypes.data,
                constructed.ctypes.data,
                stratum.ctypes.data,
                entity.ctypes.data,
                parameters.ctypes.data,
                deviation.ctypes.data,
            ),
            "tet_mesh_construct_edge_split",
        )
        if not constructed[0]:
            return None
        point.setflags(write=False)
        parameters.setflags(write=False)
        return TetMeshEdgeConstruction(
            point,
            float(parameter[0]),
            int(stratum[0]),
            int(entity[0]),
            parameters,
            float(deviation[0]),
        )

    def source_evidence(self) -> TetMeshSourceEvidence:
        """Declared and achieved source bounds, ancestry, witnesses and refusals."""
        counts = self._counts()
        vertices = int(counts[0])
        values = np.empty((2,), dtype=np.float64)
        contributor_count = np.zeros((1,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_source_evidence"](
                self._live(),
                0,
                contributor_count.ctypes.data,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            "tet_mesh_source_evidence",
        )
        offsets = np.empty((int(counts[2]) + 1,), dtype=np.int64)
        rows = np.empty((int(contributor_count[0]),), dtype=np.int32)
        segments = np.empty((int(counts[3]),), dtype=np.int32)
        strata = np.empty((vertices,), dtype=np.int8)
        entities = np.empty((vertices,), dtype=np.int32)
        parameters = np.empty((vertices, 2), dtype=np.float64)
        deviations = np.empty((vertices,), dtype=np.float64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_source_evidence"](
                self._live(),
                rows.size,
                contributor_count.ctypes.data,
                values.ctypes.data,
                offsets.ctypes.data,
                rows.ctypes.data,
                segments.ctypes.data,
                strata.ctypes.data,
                entities.ctypes.data,
                parameters.ctypes.data,
                deviations.ctypes.data,
            ),
            "tet_mesh_source_evidence",
        )
        refused = np.zeros((1,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_source_refusals"](
                self._live(),
                0,
                refused.ctypes.data,
                None,
                None,
                None,
                None,
            ),
            "tet_mesh_source_refusals",
        )
        size = int(refused[0])
        edges = np.empty((size, 2), dtype=np.int32)
        refused_strata = np.empty((size,), dtype=np.int8)
        refused_entities = np.empty((size,), dtype=np.int32)
        refused_values = np.empty((size, 2), dtype=np.float64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_source_refusals"](
                self._live(),
                size,
                refused.ctypes.data,
                edges.ctypes.data,
                refused_strata.ctypes.data,
                refused_entities.ctypes.data,
                refused_values.ctypes.data,
            ),
            "tet_mesh_source_refusals",
        )
        arrays = (
            offsets,
            rows,
            segments,
            strata,
            entities,
            parameters,
            deviations,
            edges,
            refused_strata,
            refused_entities,
            refused_values,
        )
        for array in arrays:
            array.setflags(write=False)
        return TetMeshSourceEvidence(
            float(values[0]),
            float(values[1]),
            offsets,
            rows,
            segments,
            strata,
            entities,
            parameters,
            deviations,
            edges,
            refused_strata,
            refused_entities,
            refused_values[:, 0],
            refused_values[:, 1],
            self._face_source_ids,
            self._segment_source_ids,
        )

    def measure_execution(self, enabled: bool, /) -> None:
        """Enable execution-only native timers; disabling clears their availability."""
        if not isinstance(enabled, (bool, np.bool_)):
            raise TypeError(
                f"_meshcore.TetMesh3D.measure_execution.enabled must be boolean; got {_input_summary(enabled)}."
            )
        _raise_for_call(
            self._library["phx_mc_tet_mesh_measure_execution"](
                self._live(), int(enabled)
            ),
            "tet_mesh_measure_execution",
        )

    def execution_times(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
        """Latest refinement/improvement/exudation intervals and measured flags."""
        seconds = np.empty((3,), dtype=np.float64)
        measured = np.empty((3,), dtype=np.int32)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_execution_times"](
                self._live(), seconds.ctypes.data, measured.ctypes.data
            ),
            "tet_mesh_execution_times",
        )
        return seconds, measured

    def collapse_edge(
        self,
        remove: int,
        keep: int,
        /,
        *,
        require_improvement: bool = False,
        work_limit: int | None = None,
    ) -> bool:
        """Directional collapse with full-link/source checks and atomic refusal."""
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_collapse_edge"](
            self._live(),
            _node_id(remove, "remove"),
            _node_id(keep, "keep"),
            _boolean_flag(require_improvement, "require_improvement"),
            _budget(work_limit, "work_limit"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_collapse_edge")

    def accepted_generation(self) -> int:
        """Current accepted-state generation; every committed edit advances it."""
        generation = np.zeros((1,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_accepted_generation"](
                self._live(), generation.ctypes.data
            ),
            "tet_mesh_accepted_generation",
        )
        return int(generation[0])

    def inspect_curved_collapse(
        self,
        remove: int,
        keep: int,
        /,
        *,
        maximum_faces: int,
        require_improvement: bool = False,
        work_limit: int | None = None,
    ) -> TetMeshCurvedCollapsePlan | None:
        """Validate a curved-source surface collapse without mutation.

        Native proves the volume/surface link conditions, positive cells, one
        material stratum of one curved source and exact cancellation of the
        oriented patch boundary chain. None is a refusal. ``maximum_faces``
        bounds the reported new faces (more raises a capacity error).
        """
        capacity = _budget(maximum_faces, "maximum_faces")
        if capacity < 1:
            raise ValueError(
                f"TetMesh3D.maximum_faces must be positive; got {maximum_faces}."
            )
        faces = np.empty((capacity, 3), dtype=np.int32)
        count = np.zeros((1,), dtype=np.int64)
        source = np.full((1,), -1, dtype=np.int32)
        generation = np.full((1,), -1, dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_inspect_curved_collapse"](
                self._live(),
                _node_id(remove, "remove"),
                _node_id(keep, "keep"),
                _boolean_flag(require_improvement, "require_improvement"),
                _budget(work_limit, "work_limit"),
                capacity,
                count.ctypes.data,
                source.ctypes.data,
                generation.ctypes.data,
                faces.ctypes.data,
            ),
            "tet_mesh_inspect_curved_collapse",
        )
        size = int(count[0])
        if size == 0:
            return None
        if size < 0 or size > capacity or not 0 <= source[0] < self._face_source_ids.size:
            raise RuntimeError(
                f"_meshcore.TetMesh3D.inspect_curved_collapse.remove must satisfy the input contract (Native curved inspection exceeded its reported extent.); got {_input_summary(remove)}."
            )
        counts = np.zeros((2,), dtype=np.int64)
        cavity = self._library["phx_mc_tet_mesh_curved_collapse_cavity"]
        arguments = (self._live(), int(remove), int(keep), int(generation[0]))
        _raise_for_call(
            cavity(*arguments, 0, 0, _budget(work_limit, "work_limit"),
                   None, None, counts.ctypes.data),
            "tet_mesh_curved_collapse_cavity",
        )
        old_count, new_count = (int(value) for value in counts)
        if old_count <= 0 or new_count <= 0:
            raise RuntimeError("Native curved collapse returned an empty finite cavity.")
        budget = current_native_execution_scope()
        if budget is None:
            removed = np.empty((old_count, 4), dtype=np.int32)
            proposed = np.empty((new_count, 4), dtype=np.int32)
        else:
            removed = budget.allocate_host_array((old_count, 4), np.int32)
            proposed = budget.allocate_host_array((new_count, 4), np.int32)
        _raise_for_call(
            cavity(*arguments, old_count, new_count, _budget(work_limit, "work_limit"),
                   removed.ctypes.data, proposed.ctypes.data, counts.ctypes.data),
            "tet_mesh_curved_collapse_cavity",
        )
        removed.setflags(write=False)
        proposed.setflags(write=False)
        # Native orients from the first material side; publish outward like arrays().
        outward = np.ascontiguousarray(faces[:size][:, (0, 2, 1)])
        outward.setflags(write=False)
        return TetMeshCurvedCollapsePlan(
            self,
            int(remove),
            int(keep),
            int(self._face_source_ids[int(source[0])]),
            int(source[0]),
            int(generation[0]),
            bool(require_improvement),
            outward,
            removed,
            proposed,
        )

    @checked
    def collapse_curved_edge(
        self,
        plan: TetMeshCurvedCollapsePlan,
        coordinates: ArrayLike,
        deviations: ArrayLike,
        /,
        *,
        digest: bytes,
        work_limit: int | None = None,
    ) -> bool:
        """Commit an inspected curved collapse under host whole-facet claims.

        ``coordinates`` (k, 3, 3) are the bitwise current corners of
        ``plan.faces`` and ``deviations`` (k,) their host-certified whole-facet
        deviations; ``digest`` is the source's declared 32-byte binding. Native
        re-runs every inspection proof and refuses stale generations, changed
        faces/coordinates, foreign digests or out-of-tolerance claims with the
        mesh unchanged. Committed faces become delegated-claim rows.
        """
        if plan._owner is not self:
            raise ValueError(
                f"_meshcore.TetMesh3D.collapse_curved_edge.plan must belong to this native owner; got {_input_summary(plan)}."
            )
        corners = np.ascontiguousarray(_real_array(coordinates, "coordinates"))
        claims = np.ascontiguousarray(_real_array(deviations, "deviations"))
        count = plan.faces.shape[0]
        if corners.shape != (count, 3, 3) or claims.shape != (count,):
            raise ValueError(
                f"_meshcore.TetMesh3D.collapse_curved_edge.coordinates must align with the plan faces as ({count}, 3, 3) and ({count},); got {_input_summary(coordinates)}."
            )
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise ValueError(
                f"_meshcore.TetMesh3D.collapse_curved_edge.digest must be 32 bytes; got {_input_summary(digest)}."
            )
        binding = np.frombuffer(digest, dtype=np.uint8).copy()
        faces = np.ascontiguousarray(plan.faces)
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_collapse_curved_edge"](
            ctypes.byref(
                _TetMeshCurvedCollapseRequest(
                    mesh=self._live(),
                    remove=plan.remove,
                    keep=plan.keep,
                    source=plan.source_row,
                    accepted_generation=plan.accepted_generation,
                    digest=binding.ctypes.data,
                    claim_count=count,
                    claim_faces=faces.ctypes.data,
                    claim_coordinates=corners.ctypes.data,
                    claim_deviations=claims.ctypes.data,
                    require_improvement=int(plan.require_improvement),
                    work_limit=_budget(work_limit, "work_limit"),
                    applied=applied.ctypes.data,
                )
            )
        )
        return self._applied(status, applied, "tet_mesh_collapse_curved_edge")

    def curved_claims(self) -> TetMeshCurvedClaims:
        """Every retained initial and committed curved-source host claim."""
        count = np.zeros((1,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_curved_claims"](
                self._live(), 0, count.ctypes.data, None, None, None, None, None
            ),
            "tet_mesh_curved_claims",
        )
        size = int(count[0])
        rows = np.empty((size,), dtype=np.int32)
        sources = np.empty((size,), dtype=np.int32)
        coordinates = np.empty((size, 3, 3), dtype=np.float64)
        deviations = np.empty((size,), dtype=np.float64)
        generations = np.empty((size,), dtype=np.int64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_curved_claims"](
                self._live(),
                size,
                count.ctypes.data,
                rows.ctypes.data,
                sources.ctypes.data,
                coordinates.ctypes.data,
                deviations.ctypes.data,
                generations.ctypes.data,
            ),
            "tet_mesh_curved_claims",
        )
        if (
            int(count[0]) != size
            or np.any(sources < 0)
            or np.any(sources >= self._face_source_ids.size)
        ):
            raise RuntimeError(
                f"_meshcore.TetMesh3D.curved_claims.self must satisfy the input contract (Native curved claims changed between count and copy.); got {_input_summary(self)}."
            )
        source_ids = np.ascontiguousarray(self._face_source_ids[sources])
        for array in (rows, source_ids, coordinates, deviations, generations):
            array.setflags(write=False)
        return TetMeshCurvedClaims(rows, source_ids, coordinates, deviations, generations)

    def reconnect(
        self,
        removed: object,
        proposed: object,
        /,
        *,
        work_limit: int | None = None,
    ) -> bool:
        """Replace a metric-selected cavity while preserving its oriented boundary."""
        old = _power_indices(removed, "removed")
        new = _power_indices(proposed, "proposed")
        if any(
            array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0
            for array in (old, new)
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.reconnect.removed must be nonempty tetrahedral vertex tuples; got {_input_summary(removed)}."
            )
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_reconnect"](
            self._live(),
            old.shape[0],
            old.ctypes.data,
            new.shape[0],
            new.ctypes.data,
            _budget(work_limit, "work_limit"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_reconnect")

    def remove_multiface(
        self,
        cells: object,
        apex: int,
        /,
        *,
        require_improvement: bool = False,
        work_limit: int | None = None,
    ) -> bool:
        """Reconnect a cavity identified by tetrahedral vertex tuples."""
        tuples = _power_indices(cells, "cells")
        if tuples.ndim != 2 or tuples.shape[1] != 4 or tuples.shape[0] == 0:
            raise ValueError(
                f"_meshcore.TetMesh3D.remove_multiface.cells must be a nonempty array of four-vertex tuples; got {_input_summary(cells)}."
            )
        applied = np.zeros((1,), dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_remove_multiface"](
            self._live(),
            tuples.shape[0],
            tuples.ctypes.data,
            _node_id(apex, "apex"),
            _boolean_flag(require_improvement, "require_improvement"),
            _budget(work_limit, "work_limit"),
            applied.ctypes.data,
        )
        return self._applied(status, applied, "tet_mesh_remove_multiface")

    def exude(
        self,
        *,
        min_dihedral_degrees: float = 10.0,
        max_weight_fraction: float = 0.1,
        radius_edge_bound: float = 2.0,
        minimum_relative_determinant: float = 0.0,
        max_passes: int = 8,
        work_limit: int | None = None,
    ) -> tuple[TetMeshRun[TetMeshExudeCounters], NDArray[np.float64]]:
        """Explicit bounded local weighted sliver stage, not global regularity.

        Only unprotected interior vertices receive power weights below
        ``max_weight_fraction`` times their shortest incident edge squared;
        weighted 2-3 and ring reconnections must raise the local smallest
        dihedral angle and keep size targets. No reconnection creates a cell at
        or below ``minimum_relative_determinant`` on its chart (the
        publishing validity floor); remaining cells are :meth:`unmet` records.
        Counters follow :class:`TetMeshExudeCounters`; the weights are the
        accepted power weights per native vertex.
        """
        if (
            not np.isfinite(min_dihedral_degrees)
            or not 0.0 <= min_dihedral_degrees < 70.5
            or not np.isfinite(max_weight_fraction)
            or max_weight_fraction <= 0.0
            or not np.isfinite(radius_edge_bound)
            or radius_edge_bound <= 0.0
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.exude.min_dihedral_degrees must satisfy the input contract (weighted improvement requires finite admissible quality bounds.); got {_input_summary(min_dihedral_degrees)}."
            )
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError(
                f"TetMesh3D.minimum_relative_determinant must lie in [0, 1); got {minimum_relative_determinant}."
            )
        passes = _budget(max_passes, "max_passes")
        if passes > np.iinfo(np.int32).max:
            raise ValueError(
                f"_meshcore.TetMesh3D.exude.max_passes must fit signed int32; got {_input_summary(max_passes)}."
            )
        weights = np.empty((int(self._counts()[0]),), dtype=np.float64)
        counters = np.zeros((6,), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_exude"](
            self._live(),
            min_dihedral_degrees,
            max_weight_fraction,
            radius_edge_bound,
            minimum_relative_determinant,
            passes,
            _budget(work_limit, "work_limit"),
            weights.ctypes.data,
            counters.ctypes.data,
        )
        return TetMeshRun(
            _run_status(status, "tet_mesh_exude"),
            TetMeshExudeCounters.from_slots(counters),
        ), weights

    def quality(self, *, sliver_degrees: float = 10.0) -> TetMeshQuality:
        """Dihedral/radius-edge extremes, volumes and the dihedral distribution."""

        values = np.zeros((7,), dtype=np.float64)
        histogram = np.zeros((_TET_MESH_QUALITY_BINS,), dtype=np.int64)
        slivers = np.zeros((1,), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_quality"](
            self._live(),
            sliver_degrees,
            values.ctypes.data,
            histogram.ctypes.data,
            slivers.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_quality")
        return TetMeshQuality(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]),
            histogram,
            int(slivers[0]),
        )

    def proposal_shape(
        self,
        points: ArrayLike,
        tetrahedra: ArrayLike,
        /,
        *,
        vertex_ids: ArrayLike,
        minimum_relative_determinant: float = 0.0,
        work_limit: int | None = None,
    ) -> tuple[float, float, int]:
        """Return ``(maximum_radius_edge, minimum_dihedral_degrees, below_floor)``.

        Borrow ``points`` (N, 3) and ``tetrahedra`` (T, 4) for the same native
        circumradius/shortest-edge and dihedral constructions as :meth:`quality`,
        without rebuilding a mesh, source labels or changing owned geometry.
        Empty cells return ``(0.0, 180.0, 0)``; degenerate/nonrepresentable shape
        has a nonfinite radius-edge ratio, not an admissible regular-cell fallback.

        ``vertex_ids`` (N,) are the nonnegative int64 scientific vertex
        identities of the borrowed points, including the identity a proposed
        vertex will receive; they alone choose each cell's chart.
        ``below_floor`` counts cells whose determinant does not exceed
        ``minimum_relative_determinant`` times the product of the edge lengths
        from the cell's (smallest-identity) vertex: the publishing
        :class:`~phydrax.discretization.CellValidityPolicy` floor semantics.

        Coordinate-validation visits, index visits, cell shape and floor
        evaluations consume the owner's cumulative :meth:`work_units`.
        ``work_limit`` is an additional per-query allowance (``None`` is
        unlimited), without resetting the allowance from :meth:`set_work_limit`.
        Native resource refusal raises :class:`MeshcoreError` and leaves the
        owner usable. Contiguous float64/int32 points and rows and int64
        identities are borrowed without a native copy.
        """
        point_array = _points(points, "points", 3)
        cells = _rows(tetrahedra, "tetrahedra", 4)
        raw_identities = _index_array(vertex_ids, "vertex_ids")
        if raw_identities.shape != (point_array.shape[0],):
            raise ValueError(
                f"_meshcore.TetMesh3D.proposal_shape.vertex_ids must have one identity per point; got {_input_summary(vertex_ids)}."
            )
        if np.any(raw_identities < 0) or (
            raw_identities.size and np.max(raw_identities) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                f"_meshcore.TetMesh3D.proposal_shape.vertex_ids must be nonnegative int64 identities; got {_input_summary(vertex_ids)}."
            )
        identities = np.ascontiguousarray(raw_identities, dtype=np.int64)
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError(
                f"TetMesh3D.minimum_relative_determinant must lie in [0, 1); got {minimum_relative_determinant}."
            )
        values = np.empty((3,), dtype=np.float64)
        status = self._library["phx_mc_tet_mesh_proposal_shape"](
            self._live(),
            point_array.shape[0],
            point_array.ctypes.data,
            identities.ctypes.data,
            cells.shape[0],
            cells.ctypes.data,
            minimum_relative_determinant,
            _budget(work_limit, "work_limit"),
            values.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_proposal_shape")
        return float(values[0]), float(values[1]), int(values[2])

    def unmet(self) -> TetMeshUnmet:
        """Cells left unmet by the last refinement or improvement run."""

        count = int(self._counts()[4])
        tetrahedra = np.zeros((count, 4), dtype=np.int32)
        codes = np.zeros((count, 2), dtype=np.int32)
        values = np.zeros((count,), dtype=np.float64)
        status = self._library["phx_mc_tet_mesh_unmet"](
            self._live(), tetrahedra.ctypes.data, codes.ctypes.data, values.ctypes.data
        )
        _raise_for_call(status, "tet_mesh_unmet")
        return TetMeshUnmet(tetrahedra, codes[:, 0].copy(), codes[:, 1].copy(), values)

    def arrays(self) -> TetMeshArrays:
        """Host snapshot; the mesh stays usable."""

        vertices, cells, faces, segments = (int(value) for value in self._counts()[:4])
        result = TetMeshArrays(
            np.zeros((vertices, 3), dtype=np.float64),
            np.zeros((cells, 4), dtype=np.int32),
            np.zeros((cells,), dtype=np.int32),
            np.zeros((faces, 3), dtype=np.int32),
            np.zeros((faces,), dtype=np.int32),
            np.zeros((segments, 2), dtype=np.int32),
            np.zeros((segments,), dtype=np.int32),
            np.zeros((vertices,), dtype=np.int8),
            np.zeros((vertices,), dtype=np.float64),
        )
        status = self._library["phx_mc_tet_mesh_export"](
            self._live(),
            result.points.ctypes.data,
            result.tetrahedra.ctypes.data,
            result.tetrahedron_regions.ctypes.data,
            result.faces.ctypes.data,
            result.face_sources.ctypes.data,
            result.segments.ctypes.data,
            result.segment_sources.ctypes.data,
            result.vertex_dimension.ctypes.data,
            result.vertex_sizes.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_export")
        return TetMeshArrays(
            result.points,
            result.tetrahedra,
            result.tetrahedron_regions,
            result.faces,
            self._face_source_ids[result.face_sources],
            result.segments,
            self._segment_source_ids[result.segment_sources],
            result.vertex_dimension,
            result.vertex_sizes,
        )


@final
class RestrictedPowerOutputLimits(StrictModule):
    """Original output allowances, checked natively before publication.

    Face/cell entries are occurrences, not distinct entities. The connectivity
    and numeric-data bounds govern the explicit-loop topology owner; native
    construction payloads instead consume the active managed-storage scope.
    Query and scratch limits remain owned by that same request scope.
    Nonnegative authored allowances above the signed ABI range saturate at
    INT64_MAX: every representable native count is still admitted.
    """

    max_edges: int = eqx.field(static=True, default=2**63 - 1)
    max_faces: int = eqx.field(static=True, default=2**63 - 1)
    max_face_entries: int = eqx.field(static=True, default=2**63 - 1)
    max_cells: int = eqx.field(static=True, default=2**63 - 1)
    max_site_witnesses: int = eqx.field(static=True, default=2**63 - 1)
    max_sided_loop_occurrences: int = eqx.field(static=True, default=2**63 - 1)
    max_sided_corner_occurrences: int = eqx.field(static=True, default=2**63 - 1)
    max_connectivity_entries: int = eqx.field(static=True, default=2**63 - 1)
    max_data_bytes: int = eqx.field(static=True, default=2**63 - 1)

    def native_fields(self) -> dict[str, int]:
        return {
            name: min(nonnegative_integer(value, name), 2**63 - 1)
            for name, value in vars(self).items()
        }


@final
class RestrictedPowerPreflight(StrictModule):
    """Exact native topology counts and the existing topology-owner envelope."""

    vertices: int = eqx.field(static=True)
    edges: int = eqx.field(static=True)
    faces: int = eqx.field(static=True)
    face_corners: int = eqx.field(static=True)
    cells: int = eqx.field(static=True)
    pieces: int = eqx.field(static=True)
    vertex_site_witnesses: int = eqx.field(static=True)
    sided_loop_occurrences: int = eqx.field(static=True)
    sided_corner_occurrences: int = eqx.field(static=True)
    native_output_payload_bytes: int = eqx.field(static=True)
    publication_connectivity_entries_upper: int = eqx.field(static=True)
    publication_numeric_data_bytes_upper: int = eqx.field(static=True)

    @classmethod
    def from_slots(cls, slots: NDArray[np.int64], /) -> RestrictedPowerPreflight:
        if slots.shape != (12,) or slots.dtype != np.dtype(np.int64):
            raise ValueError("RestrictedPowerPreflight.slots must be twelve int64 counts")
        return cls(*(int(value) for value in slots))

    def items(self) -> tuple[tuple[str, int], ...]:
        return tuple(vars(self).items())


@final
class RestrictedPowerCapacity(StrictModule):
    """One exact failed output allowance, in deterministic native axis order."""

    axis: str = eqx.field(static=True)
    required: int = eqx.field(static=True)
    remaining: int = eqx.field(static=True)


def _power_preflight(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> RestrictedPowerPreflight:
    slots = np.zeros(12, dtype=np.int64)
    _raise_for_call(
        library["phx_mc_power_cells_preflight"](handle, slots.ctypes.data),
        "phx_mc_power_cells_preflight",
    )
    return RestrictedPowerPreflight.from_slots(slots)


@final
class RestrictedPowerCells(StrictModule):
    """Native restricted power cells and decomposition lineage on the host.

    Face loops point outward from the first cell; a negative second cell encodes
    ``-(1 + boundary_facet)``. Second moments are scalar integrals of squared
    distance to the site, not full covariance tensors.
    """

    vertices: NDArray[np.float64]
    face_offsets: NDArray[np.int64]
    face_vertices: NDArray[np.int32]
    face_cells: NDArray[np.int32]
    face_facets: NDArray[np.int32]
    cell_sites: NDArray[np.int32]
    cell_regions: NDArray[np.int32]
    cell_volumes: NDArray[np.float64]
    cell_moments: NDArray[np.float64]
    cell_second_moments: NDArray[np.float64]
    piece_sites: NDArray[np.int32]
    piece_tets: NDArray[np.int32]
    piece_cells: NDArray[np.int32]
    piece_volumes: NDArray[np.float64]
    counters: RestrictedPowerCounters
    vertex_carriers: NDArray[np.int32]
    vertex_site_offsets: NDArray[np.int64]
    vertex_sites: NDArray[np.int32]
    source_work_units: int = eqx.field(static=True)
    preflight: RestrictedPowerPreflight


@final
class RestrictedPowerFailure(MeshcoreError):
    """Native refusal retaining the failure location and consumed work."""

    def __init__(
        self,
        status: MeshcoreStatus,
        failure: NDArray[np.int64],
        counters: NDArray[np.int64],
        /,
        *,
        preflight: RestrictedPowerPreflight,
        capacity: RestrictedPowerCapacity | None,
    ) -> None:
        detail = "" if capacity is None else (
            f"; {capacity.axis}: required={capacity.required}, remaining={capacity.remaining}"
        )
        super().__init__(status, f"restricted_power_cells: {tuple(failure)}{detail}")
        self.failure = failure
        self.counters = RestrictedPowerCounters.from_slots(counters)
        self.preflight = preflight
        self.capacity = capacity


def _power_indices(value: object, name: str, /) -> NDArray[np.int32]:
    values = _index_array(value, name)
    if np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max):
        raise ValueError(
            f"{'_meshcore.' + name} must fit signed int32; got {_input_summary(value)}."
        )
    return np.ascontiguousarray(values, dtype=np.int32)


def _copy_power_cells(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    counters: NDArray[np.int64],
    /,
    *,
    admit_host_output: Callable[[int], None] | None = None,
) -> RestrictedPowerCells:
    sizes = np.zeros((5,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_power_cells_sizes"](handle, sizes.ctypes.data),
        "phx_mc_power_cells_sizes",
    )
    source_work_units = library["phx_mc_power_cells_source_work_units"](handle)
    if source_work_units < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._copy_power_cells.library must satisfy the input contract (negative power-cell source work); got {_input_summary(library)}.",
        )
    if np.any(sizes < 0):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._copy_power_cells.library must satisfy the input contract (negative power-cell sizes); got {_input_summary(library)}.",
        )
    vertices, faces, entries, cells, pieces = (int(size) for size in sizes)
    witness_count = int(library["phx_mc_power_cells_site_witness_size"](handle))
    if witness_count < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._copy_power_cells.library must satisfy the input contract (negative power-cell site construction record size); got {_input_summary(library)}.",
        )
    preflight = _power_preflight(library, handle)
    output_bytes = (
        48 * vertices + 20 * faces + 4 * entries
        + 48 * cells + 20 * pieces + 4 * witness_count + 16
    )
    if (
        (vertices, faces, entries, cells, pieces, witness_count)
        != (
            preflight.vertices, preflight.faces, preflight.face_corners,
            preflight.cells, preflight.pieces, preflight.vertex_site_witnesses,
        )
        or preflight.native_output_payload_bytes != output_bytes
    ):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            "restricted_power_cells.preflight must match the admitted export payload",
        )
    if admit_host_output is not None:
        # Seventeen arrays, result/counter/preflight records, and twenty-three
        # int64 scalar owners. Native payload stays live in the same managed cap.
        admit_host_output(
            20 * 256 + 23 * (128 + 4 * 8) + output_bytes
        )
    result = RestrictedPowerCells(
        np.empty((vertices, 3), dtype=np.float64),
        np.empty((faces + 1,), dtype=np.int64),
        np.empty((entries,), dtype=np.int32),
        np.empty((faces, 2), dtype=np.int32),
        np.empty((faces,), dtype=np.int32),
        np.empty((cells,), dtype=np.int32),
        np.empty((cells,), dtype=np.int32),
        np.empty((cells,), dtype=np.float64),
        np.empty((cells, 3), dtype=np.float64),
        np.empty((cells,), dtype=np.float64),
        np.empty((pieces,), dtype=np.int32),
        np.empty((pieces,), dtype=np.int32),
        np.empty((pieces,), dtype=np.int32),
        np.empty((pieces,), dtype=np.float64),
        RestrictedPowerCounters.from_slots(counters),
        np.empty((vertices, 4), dtype=np.int32),
        np.empty((vertices + 1,), dtype=np.int64),
        np.empty((witness_count,), dtype=np.int32),
        source_work_units,
        preflight,
    )
    _raise_for_call(
        library["phx_mc_power_cells_export"](
            handle,
            result.vertices.ctypes.data,
            result.face_offsets.ctypes.data,
            result.face_vertices.ctypes.data,
            result.face_cells.ctypes.data,
            result.face_facets.ctypes.data,
            result.cell_sites.ctypes.data,
            result.cell_regions.ctypes.data,
            result.cell_volumes.ctypes.data,
            result.cell_moments.ctypes.data,
            result.cell_second_moments.ctypes.data,
            result.piece_sites.ctypes.data,
            result.piece_tets.ctypes.data,
            result.piece_cells.ctypes.data,
            result.piece_volumes.ctypes.data,
        ),
        "phx_mc_power_cells_export",
    )
    _raise_for_call(
        library["phx_mc_power_cells_export_vertex_carriers"](
            handle, result.vertex_carriers.ctypes.data
        ),
        "phx_mc_power_cells_export_vertex_carriers",
    )
    result.vertex_carriers.flags.writeable = False
    _raise_for_call(
        library["phx_mc_power_cells_export_vertex_sites"](
            handle,
            result.vertex_site_offsets.ctypes.data,
            result.vertex_sites.ctypes.data,
        ),
        "phx_mc_power_cells_export_vertex_sites",
    )
    result.vertex_site_offsets.flags.writeable = False
    result.vertex_sites.flags.writeable = False
    return result


def restricted_power_cells(
    sites: ArrayLike,
    weights: ArrayLike,
    neighbor_offsets: ArrayLike,
    neighbors: ArrayLike,
    split_sites: ArrayLike,
    points: ArrayLike,
    tets: ArrayLike,
    tet_regions: ArrayLike,
    tet_face_facets: ArrayLike,
    /,
    *,
    max_pieces: int,
    max_vertices: int,
    work_limit: int,
    record_native_phase: NativePhaseRecorder | None = None,
    admit_host_output: Callable[[int], None] | None = None,
    output_limits: RestrictedPowerOutputLimits | None = None,
) -> RestrictedPowerCells:
    """Clip weighted cells to a declared tetrahedral domain.

    Neighbor CSR comes from the native regular triangulation. Domain cells
    must be positively oriented with every exterior face explicitly tagged.
    A native failure never returns partial cells as an accepted decomposition.
    """

    native_limits = (output_limits or RestrictedPowerOutputLimits()).native_fields()
    site_array = _points(sites, "sites", 3)
    point_array = _points(points, "points", 3)
    weight_array = _real_array(weights, "weights")
    offsets = np.ascontiguousarray(
        _index_array(neighbor_offsets, "neighbor_offsets"), dtype=np.int64
    )
    adjacent = _power_indices(neighbors, "neighbors")
    split = _index_array(split_sites, "split_sites")
    tetrahedra = _power_indices(tets, "tets")
    regions = _power_indices(tet_regions, "tet_regions")
    facets = _power_indices(tet_face_facets, "tet_face_facets")
    count = site_array.shape[0]
    if (
        count == 0
        or weight_array.shape != (count,)
        or not np.all(np.isfinite(weight_array))
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells.weights must be a finite vector with one value per site; got {_input_summary(weights)}."
        )
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or adjacent.ndim != 1
        or offsets[-1] != adjacent.size
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(adjacent < 0)
        or np.any(adjacent >= count)
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells.neighbor_offsets must define site-indexed CSR; got {_input_summary(neighbor_offsets)}."
        )
    if split.shape != (count,) or np.any((split != 0) & (split != 1)):
        raise ValueError(
            f"_meshcore.restricted_power_cells.sites must contain one zero-or-one flag per site; got {_input_summary(sites)}."
        )
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or regions.shape != (tetrahedra.shape[0],)
        or facets.shape != tetrahedra.shape
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= point_array.shape[0])
    ):
        raise ValueError(
            f"restricted_power_cells.tets must align tet_regions and tet_face_facets with domain points; got {_input_summary(tetrahedra)}."
        )
    flags = np.ascontiguousarray(split, dtype=np.int8)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_restricted_power_cells"](
            ctypes.byref(
                _RestrictedPowerCellsRequest(
                    site_count=count,
                    sites=site_array.ctypes.data,
                    weights=weight_array.ctypes.data,
                    neighbor_offsets=offsets.ctypes.data,
                    neighbors=adjacent.ctypes.data,
                    split_sites=flags.ctypes.data,
                    point_count=point_array.shape[0],
                    points=point_array.ctypes.data,
                    tet_count=tetrahedra.shape[0],
                    tets=tetrahedra.ctypes.data,
                    tet_regions=regions.ctypes.data,
                    tet_face_facets=facets.ctypes.data,
                    max_pieces=_limit(max_pieces, 1, "max_pieces"),
                    max_vertices=_limit(max_vertices, 1, "max_vertices"),
                    work_limit=_limit(work_limit, 1, "work_limit"),
                    record_phases=int(record_native_phase is not None),
                    result=ctypes.addressof(handle),
                    **native_limits,
                )
            )
        )
    )
    return _finish_power_cells(
        library,
        handle,
        status,
        record_native_phase,
        admit_host_output=admit_host_output,
    )


def _finish_power_cells(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    status: MeshcoreStatus,
    record_native_phase: NativePhaseRecorder | None,
    /,
    *,
    admit_host_output: Callable[[int], None] | None = None,
) -> RestrictedPowerCells:
    """Own every native export and failure record until the result is freed."""
    if not handle.value:
        raise MeshcoreError(
            status if status != MeshcoreStatus.OK else MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._finish_power_cells.library must satisfy the input contract (missing power-cell result); got {_input_summary(library)}.",
        )
    # Own the native result until all exports finish, including failure evidence.
    try:
        counters = np.zeros((10,), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_power_cells_counters"](handle, counters.ctypes.data),
            "phx_mc_power_cells_counters",
        )
        if record_native_phase is not None:
            seconds = np.empty((4,), dtype=np.float64)
            _raise_for_call(
                library["phx_mc_power_cells_phase_seconds"](handle, seconds.ctypes.data),
                "phx_mc_power_cells_phase_seconds",
            )
            if not np.all(np.isfinite(seconds)):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR,
                    f"_meshcore._finish_power_cells.library must satisfy the input contract (invalid power-cell timings); got {_input_summary(library)}.",
                )
            phases = (
                "power_adjacency",
                "power_clipping",
                "power_components",
                "power_publication",
            )
            for index, phase in enumerate(phases):
                if seconds[index] >= 0.0:
                    work = int(counters[0] + counters[3]) if index == 1 else None
                    record_native_phase(phase, float(seconds[index]), work, 1)
        if status != MeshcoreStatus.OK:
            failure = np.zeros((4,), dtype=np.int64)
            _raise_for_call(
                library["phx_mc_power_cells_failure"](handle, failure.ctypes.data),
                "phx_mc_power_cells_failure",
            )
            preflight = _power_preflight(library, handle)
            capacity_slots = np.zeros(3, dtype=np.int64)
            _raise_for_call(
                library["phx_mc_power_cells_capacity"](handle, capacity_slots.ctypes.data),
                "phx_mc_power_cells_capacity",
            )
            axis, required, remaining = (int(value) for value in capacity_slots)
            capacity = None
            if axis:
                if not 1 <= axis <= 12 or required <= remaining or remaining < 0:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "restricted_power_cells.capacity must identify an exceeded allowance",
                    )
                name, expected = preflight.items()[axis - 1]
                if required != expected:
                    raise MeshcoreError(
                        MeshcoreStatus.INTERNAL_ERROR,
                        "restricted_power_cells.capacity must match the preflight requirement",
                    )
                capacity = RestrictedPowerCapacity(name, required, remaining)
            raise RestrictedPowerFailure(
                status, failure, counters, preflight=preflight, capacity=capacity
            )
        return _copy_power_cells(
            library, handle, counters, admit_host_output=admit_host_output
        )
    finally:
        library["phx_mc_power_cells_free"](handle)


def restricted_power_cells_exact(
    sites: ArrayLike,
    weights: ArrayLike,
    image_site_owners: ArrayLike,
    image_coordinate_offsets: ArrayLike,
    image_coordinate_components: ArrayLike,
    neighbor_offsets: ArrayLike,
    neighbors: ArrayLike,
    split_sites: ArrayLike,
    points: ArrayLike,
    tets: ArrayLike,
    tet_regions: ArrayLike,
    tet_face_facets: ArrayLike,
    /,
    *,
    max_pieces: int,
    max_vertices: int,
    work_limit: int,
    record_native_phase: NativePhaseRecorder | None = None,
    admit_host_output: Callable[[int], None] | None = None,
    output_limits: RestrictedPowerOutputLimits | None = None,
) -> RestrictedPowerCells:
    """Clip cells from image coordinates and their input site owners."""
    native_limits = (output_limits or RestrictedPowerOutputLimits()).native_fields()
    original = _points(sites, "sites", 3)
    weight_array = _real_array(weights, "weights")
    owners = _power_indices(image_site_owners, "image_site_owners")
    coordinate_offsets = np.ascontiguousarray(
        _index_array(image_coordinate_offsets, "image_coordinate_offsets"),
        dtype=np.int64,
    )
    components = np.ascontiguousarray(
        _real_array(image_coordinate_components, "image_coordinate_components"),
        dtype=np.float64,
    )
    if (
        original.shape[0] == 0
        or weight_array.shape != (original.shape[0],)
        or not np.all(np.isfinite(weight_array))
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells_exact.weights must be a finite vector with one value per input site; got {_input_summary(weights)}."
        )
    if (
        owners.ndim != 1
        or owners.size == 0
        or owners.size > _MAX_POINTS
        or np.any(owners < 0)
        or np.any(owners >= original.shape[0])
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells_exact.image_site_owners must identify every image's input site; got {_input_summary(image_site_owners)}."
        )
    image_count = owners.size
    if (
        components.ndim != 1
        or not np.all(np.isfinite(components))
        or coordinate_offsets.shape != (3 * image_count + 1,)
        or coordinate_offsets[0] != 0
        or coordinate_offsets[-1] != components.size
        or np.any(coordinate_offsets[1:] < coordinate_offsets[:-1])
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells_exact.sites must bind entire finite expansion components; got {_input_summary(sites)}."
        )
    domain = _points(points, "points", 3)
    offsets = np.ascontiguousarray(
        _index_array(neighbor_offsets, "neighbor_offsets"),
        dtype=np.int64,
    )
    adjacent = _power_indices(neighbors, "neighbors")
    split = _index_array(split_sites, "split_sites")
    tetrahedra = _power_indices(tets, "tets")
    regions = _power_indices(tet_regions, "tet_regions")
    facets = _power_indices(tet_face_facets, "tet_face_facets")
    if (
        offsets.shape != (image_count + 1,)
        or offsets[0] != 0
        or adjacent.ndim != 1
        or offsets[-1] != adjacent.size
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(adjacent < 0)
        or np.any(adjacent >= image_count)
    ):
        raise ValueError(
            f"_meshcore.restricted_power_cells_exact.neighbor_offsets must define image-indexed CSR; got {_input_summary(neighbor_offsets)}."
        )
    if split.shape != (image_count,) or np.any((split != 0) & (split != 1)):
        raise ValueError(
            f"_meshcore.restricted_power_cells_exact.sites must contain one zero-or-one flag per image; got {_input_summary(sites)}."
        )
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or regions.shape != (tetrahedra.shape[0],)
        or facets.shape != tetrahedra.shape
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= domain.shape[0])
    ):
        raise ValueError(
            f"restricted_power_cells_exact.tets must align tet_regions and tet_face_facets with domain points; got {_input_summary(tetrahedra)}."
        )
    flags = np.ascontiguousarray(split, dtype=np.int8)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_restricted_power_cells_exact"](
            ctypes.byref(
                _RestrictedPowerCellsExactRequest(
                    original_site_count=original.shape[0],
                    original_sites=original.ctypes.data,
                    original_weights=weight_array.ctypes.data,
                    image_count=image_count,
                    image_site_owners=owners.ctypes.data,
                    image_coordinate_offsets=coordinate_offsets.ctypes.data,
                    component_count=components.size,
                    image_coordinate_components=components.ctypes.data,
                    neighbor_offsets=offsets.ctypes.data,
                    neighbors=adjacent.ctypes.data,
                    split_sites=flags.ctypes.data,
                    point_count=domain.shape[0],
                    points=domain.ctypes.data,
                    tet_count=tetrahedra.shape[0],
                    tets=tetrahedra.ctypes.data,
                    tet_regions=regions.ctypes.data,
                    tet_face_facets=facets.ctypes.data,
                    max_pieces=_limit(max_pieces, 1, "max_pieces"),
                    max_vertices=_limit(max_vertices, 1, "max_vertices"),
                    work_limit=_limit(work_limit, 1, "work_limit"),
                    record_phases=int(record_native_phase is not None),
                    result=ctypes.addressof(handle),
                    **native_limits,
                )
            )
        )
    )
    return _finish_power_cells(
        library,
        handle,
        status,
        record_native_phase,
        admit_host_output=admit_host_output,
    )


ARRANGEMENT_UNCLASSIFIED = int(np.iinfo(np.int32).min)


@final
class ArrangementClassification(StrictModule):
    """Operand membership of arrangement fragments.

    Fragments of one operand are joined across shared edges that are not
    contact edges. ``component_windings[c, j]`` is the winding number of
    operand ``j`` at component ``c`` (``fragment_components`` maps fragments to
    components), or ``ARRANGEMENT_UNCLASSIFIED`` for the component's own
    operand and operands coplanar with it. ``status`` is ``OK`` or
    ``CAPACITY_EXCEEDED``; a refused classification has empty arrays.
    ``pair_scans`` (representative/triangle pairs) and ``matrix_entries`` are
    the admitted work, or on refusal the requirement evaluated before it was
    refused; ``components`` is the component count.
    """

    status: MeshcoreStatus = eqx.field(static=True)
    fragment_components: NDArray[np.int64]
    component_windings: NDArray[np.int32]
    components: int = eqx.field(static=True)
    pair_scans: int = eqx.field(static=True)
    matrix_entries: int = eqx.field(static=True)


@final
class NativeArrangement(StrictModule):
    """Construction identities with bounded host-coordinate publication."""

    vertices: NDArray[np.float64]
    bounds: NDArray[np.float64]
    constructions: NDArray[np.int8]
    origins: NDArray[np.int64]
    fragments: NDArray[np.int64]
    fragment_faces: NDArray[np.int64]
    contact_edges: NDArray[np.int64]
    coplanar: NDArray[np.int64]
    locations: NDArray[np.int64]
    counters: ArrangementCounters
    classification: ArrangementClassification | None


@final
class ArrangementFailure(MeshcoreError):
    """Arrangement refusal at an input face, without a partial accepted mesh."""

    def __init__(self, status: MeshcoreStatus, failed_face: int, /) -> None:
        super().__init__(status, f"arrange_triangles: failed face {failed_face}")
        self.failed_face = failed_face


def _copy_arrangement(
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> NativeArrangement:
    sizes = np.zeros((5,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_arrangement_sizes"](handle, sizes.ctypes.data),
        "phx_mc_arrangement_sizes",
    )
    if np.any(sizes < 0):
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore._copy_arrangement.library must satisfy the input contract (negative arrangement sizes); got {_input_summary(library)}.",
        )
    vertices, fragments, edges, coplanar, locations = (int(size) for size in sizes)
    counters = np.empty((11,), dtype=np.int64)
    _raise_for_call(
        library["phx_mc_arrangement_counters"](handle, counters.ctypes.data),
        "phx_mc_arrangement_counters",
    )
    result = NativeArrangement(
        np.empty((vertices, 3), dtype=np.float64),
        np.empty((vertices,), dtype=np.float64),
        np.empty((vertices,), dtype=np.int8),
        np.empty((vertices,), dtype=np.int64),
        np.empty((fragments, 3), dtype=np.int64),
        np.empty((fragments,), dtype=np.int64),
        np.empty((edges, 2), dtype=np.int64),
        np.empty((coplanar, 3), dtype=np.int64),
        np.empty((locations, 3), dtype=np.int64),
        ArrangementCounters.from_slots(counters),
        None,
    )
    _raise_for_call(
        library["phx_mc_arrangement_export"](
            handle,
            result.vertices.ctypes.data,
            result.bounds.ctypes.data,
            result.constructions.ctypes.data,
            result.origins.ctypes.data,
            result.fragments.ctypes.data,
            result.fragment_faces.ctypes.data,
            result.contact_edges.ctypes.data,
            result.coplanar.ctypes.data,
            result.locations.ctypes.data,
        ),
        "phx_mc_arrangement_export",
    )
    return result


def _classify_arrangement(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    fragments: int,
    operand_count: int,
    work_limit: int,
    /,
) -> ArrangementClassification:
    failed = np.full((1,), -1, dtype=np.int64)
    sizes = np.zeros((4,), dtype=np.int64)
    status = MeshcoreStatus(
        library["phx_mc_arrangement_classify"](
            handle, operand_count, work_limit, failed.ctypes.data, sizes.ctypes.data
        )
    )
    components, operands, pair_scans, matrix_entries = (int(size) for size in sizes)
    match status:
        case MeshcoreStatus.OK:
            fragment_components = np.empty((fragments,), dtype=np.int64)
            component_windings = np.empty((components, operands), dtype=np.int32)
            _raise_for_call(
                library["phx_mc_arrangement_export_classification"](
                    handle,
                    fragment_components.ctypes.data,
                    component_windings.ctypes.data,
                ),
                "phx_mc_arrangement_export_classification",
            )
        case MeshcoreStatus.CAPACITY_EXCEEDED:
            fragment_components = np.empty((0,), dtype=np.int64)
            component_windings = np.empty((0, operand_count), dtype=np.int32)
        case _:
            raise ArrangementFailure(status, int(failed[0]))
    return ArrangementClassification(
        status,
        fragment_components,
        component_windings,
        components,
        pair_scans,
        matrix_entries,
    )


def arrange_triangles(
    vertices: ArrayLike,
    triangles: ArrayLike,
    surfaces: ArrayLike,
    pairs: ArrayLike,
    /,
    *,
    maximum_vertices: int,
    maximum_fragments: int,
    classification_operands: int | None = None,
    classification_work_limit: int | None = None,
) -> NativeArrangement:
    """Arrange all declared cross-operand triangle contacts simultaneously.

    Candidate pairs are explicit bounded input from the spatial route.
    The native indirect predicates retain input construction identity.
    With ``classification_operands`` (the declared dense operand count of
    ``surfaces``) and ``classification_work_limit`` the operand
    membership of every fragment component is decided on the implicit points
    as well; see :class:`ArrangementClassification`.
    """

    points = _points(vertices, "vertices", 3)
    faces = np.ascontiguousarray(_index_array(triangles, "triangles"), dtype=np.int64)
    owners = _power_indices(surfaces, "surfaces")
    candidates = np.ascontiguousarray(_index_array(pairs, "pairs"), dtype=np.int64)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or owners.shape != (faces.shape[0],)
        or candidates.ndim != 2
        or candidates.shape[1] != 2
        or np.any(faces < 0)
        or np.any(faces >= points.shape[0])
        or np.any(candidates < 0)
        or np.any(candidates >= faces.shape[0])
    ):
        raise ValueError(
            f"_meshcore.arrange_triangles.triangles must align with input vertices; got {_input_summary(triangles)}."
        )
    vertex_limit = _limit(maximum_vertices, points.shape[0], "maximum_vertices")
    fragment_limit = _limit(maximum_fragments, faces.shape[0], "maximum_fragments")
    if vertex_limit < points.shape[0] or fragment_limit < faces.shape[0]:
        raise ValueError(
            f"_meshcore.arrange_triangles.vertices must include the input geometry; got {_input_summary(vertices)}."
        )
    if (classification_operands is None) != (classification_work_limit is None):
        raise ValueError(
            f"_meshcore.arrange_triangles.classification_operands must satisfy the input contract (classification_operands and classification_work_limit are given together.); got {_input_summary(classification_operands)}."
        )
    classification_request = None
    if classification_operands is not None and classification_work_limit is not None:
        work_limit = _unsigned_budget(
            classification_work_limit, "classification_work_limit"
        )
        if work_limit > np.iinfo(np.int64).max:
            raise ValueError(
                f"_meshcore.arrange_triangles.classification_work_limit must fit a signed int64; got {_input_summary(classification_work_limit)}."
            )
        classification_request = (
            _limit(classification_operands, 1, "classification_operands"),
            work_limit,
        )
    library = load_meshcore()
    handle = ctypes.c_void_p()
    failed = np.full((1,), -1, dtype=np.int64)
    status = MeshcoreStatus(
        library["phx_mc_arrange_triangles"](
            points.shape[0],
            points.ctypes.data,
            faces.shape[0],
            faces.ctypes.data,
            owners.ctypes.data,
            candidates.shape[0],
            candidates.ctypes.data,
            vertex_limit,
            fragment_limit,
            failed.ctypes.data,
            ctypes.addressof(handle),
        )
    )
    # A failed call can own a diagnostic handle; it must be released as well.
    try:
        if status != MeshcoreStatus.OK:
            raise ArrangementFailure(status, int(failed[0]))
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.arrange_triangles.vertices must satisfy the input contract (missing arrangement result); got {_input_summary(vertices)}.",
            )
        result = _copy_arrangement(library, handle)
        if classification_request is None:
            return result
        return eqx.tree_at(
            lambda item: item.classification,
            result,
            _classify_arrangement(
                library, handle, result.fragments.shape[0], *classification_request
            ),
            is_leaf=lambda item: item is None,
        )
    finally:
        if handle.value:
            library["phx_mc_arrangement_free"](handle)


def _dual_template(
    nodes: object,
    vertex_count: int,
    maximum_cells: int,
    *,
    width: int,
    multiplier: int,
    corners: int,
    name: str,
) -> NDArray[np.int64]:
    data = np.ascontiguousarray(_index_array(nodes, "nodes"), dtype=np.int64)
    count = _limit(vertex_count, 1, "vertex_count")
    if data.ndim != 2 or data.shape[1] != width:
        raise ValueError(
            f"_meshcore._dual_template.nodes must have shape (cells, {width}); got {_input_summary(nodes)}."
        )
    if np.any(data < 0) or np.any(data >= count):
        raise ValueError(
            f"_meshcore._dual_template.nodes must index the declared vertex table; got {_input_summary(nodes)}."
        )
    capacity = _limit(maximum_cells, 1, "maximum_cells")
    needed = multiplier * data.shape[0]
    if needed > capacity:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"{'_meshcore.' + name} must satisfy the input contract ({name}: cell budget); got {_input_summary(nodes)}.",
        )
    output = np.empty((needed, corners), dtype=np.int64)
    status = load_meshcore()[name](
        data.shape[0], count, data.ctypes.data, needed, output.ctypes.data
    )
    _raise_for_call(status, name.removeprefix("phx_mc_"))
    return output


def triangle_dual_quads(
    nodes: object,
    vertex_count: int,
    /,
    *,
    maximum_cells: int,
) -> NDArray[np.int64]:
    """Three conforming dual quads per triangle from shared seven-node tuples."""
    return _dual_template(
        nodes,
        vertex_count,
        maximum_cells,
        width=7,
        multiplier=3,
        corners=4,
        name="phx_mc_triangle_dual_quads",
    )


def tetrahedron_dual_hexes(
    nodes: object,
    vertex_count: int,
    /,
    *,
    maximum_cells: int,
) -> NDArray[np.int64]:
    """Return four dual hexes per tetrahedron from shared fifteen-node tuples."""
    return _dual_template(
        nodes,
        vertex_count,
        maximum_cells,
        width=15,
        multiplier=4,
        corners=8,
        name="phx_mc_tetrahedron_dual_hexes",
    )


def subdivide_hex_grid(
    nodes: object,
    vertex_count: int,
    /,
    *,
    maximum_cells: int,
) -> NDArray[np.int64]:
    """Eight refined hexes from shared 27-node logical-grid tuples."""
    return _dual_template(
        nodes,
        vertex_count,
        maximum_cells,
        width=27,
        multiplier=8,
        corners=8,
        name="phx_mc_subdivide_hex_grid",
    )


def surface_envelope(
    vertices: ArrayLike,
    faces: ArrayLike,
    /,
    *,
    offset: float,
    spacing: float,
    certificate_spacing: float,
    max_samples: int,
    max_work_units: int,
    max_vertices: int,
    max_triangles: int,
    max_volume_vertices: int,
    max_tetrahedra: int,
) -> SurfaceEnvelope:
    """Return a bounded wrapping surface and its enclosed volume cells."""
    points = _points(vertices, "vertices", 3)
    triangles = np.ascontiguousarray(_index_array(faces, "faces"), dtype=np.int64)
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or np.any(triangles < 0)
        or np.any(triangles >= points.shape[0])
    ):
        raise ValueError(
            f"_meshcore.surface_envelope.faces must contain three valid vertex indices per triangle; got {_input_summary(faces)}."
        )
    parameters = np.asarray((offset, spacing, certificate_spacing), dtype=np.float64)
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError(
            f"_meshcore.surface_envelope.offset must be finite positive; got {_input_summary(offset)}."
        )
    capacities = (
        _limit(max_vertices, 1, "max_vertices"),
        _limit(max_triangles, 1, "max_triangles"),
        _limit(max_volume_vertices, 1, "max_volume_vertices"),
        _limit(max_tetrahedra, 1, "max_tetrahedra"),
    )
    counts = np.zeros((7,), dtype=np.int64)
    bounds = np.empty((4,), dtype=np.float64)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = library["phx_mc_surface_envelope_create"](
        ctypes.byref(
            _SurfaceEnvelopeCreateRequest(
                vertices=points.ctypes.data,
                vertex_count=points.shape[0],
                faces=triangles.ctypes.data,
                face_count=triangles.shape[0],
                offset=float(parameters[0]),
                spacing=float(parameters[1]),
                certificate_spacing=float(parameters[2]),
                sample_limit=_limit(max_samples, 1, "max_samples"),
                work_limit=_limit(max_work_units, 1, "max_work_units"),
                surface_vertex_capacity=capacities[0],
                surface_triangle_capacity=capacities[1],
                volume_vertex_capacity=capacities[2],
                tetrahedron_capacity=capacities[3],
                result=ctypes.addressof(handle),
                counts=counts.ctypes.data,
                bounds=bounds.ctypes.data,
            )
        )
    )
    try:
        if status == MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                f"surface_envelope.bounds must admit finite source inclusion at the declared scales; "
                f"got points={_input_summary(points)}, offset={offset}, spacing={spacing}, certificate_spacing={certificate_spacing}."
            )
        _raise_for_call(status, "surface_envelope")
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.surface_envelope.vertices must satisfy the input contract (missing envelope result); got {_input_summary(vertices)}.",
            )
        sizes = (int(counts[0]), int(counts[1]), int(counts[5]), int(counts[6]))
        if any(
            size < 0 or size > limit
            for size, limit in zip(sizes, capacities, strict=True)
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.surface_envelope.vertices must satisfy the input contract (invalid envelope output counts); got {_input_summary(vertices)}.",
            )
        surface_points = np.empty((sizes[0], 3), dtype=np.float64)
        surface_faces = np.empty((sizes[1], 3), dtype=np.int64)
        volume_points = np.empty((sizes[2], 3), dtype=np.float64)
        cells = np.empty((sizes[3], 4), dtype=np.int64)
        _raise_for_call(
            library["phx_mc_surface_envelope_arrays"](
                handle,
                surface_points.ctypes.data,
                surface_faces.ctypes.data,
                volume_points.ctypes.data,
                cells.ctypes.data,
            ),
            "surface_envelope_arrays",
        )
        return SurfaceEnvelope(
            surface_points=surface_points,
            surface_faces=surface_faces,
            counts=SurfaceEnvelopeCounts.from_slots(counts),
            bounds=bounds,
            volume_points=volume_points,
            cells=cells,
        )
    finally:
        if handle.value:
            library["phx_mc_surface_envelope_free"](handle)


def restricted_dual_3d(
    points: ArrayLike,
    tets: ArrayLike,
    domain: ArrayLike,
    max_facets: int,
    work_limit: int,
    /,
) -> RestrictedDual3D:
    """Numerical dual workset; item status is not a root/coverage certificate."""
    coordinates = _points(points, "points", 3)
    cells = _power_indices(tets, "tets")
    bounds = _real_array(domain, "domain")
    if (
        cells.ndim != 2
        or cells.shape[1] != 4
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
        or bounds.shape not in ((6,), (2, 3))
    ):
        raise ValueError(
            f"_meshcore.restricted_dual_3d.tets must describe a three-dimensional point set; got {_input_summary(tets)}."
        )
    bounds = np.ascontiguousarray(bounds.reshape((6,)), dtype=np.float64)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError(
            f"restricted_dual_3d.domain must be finite and ordered; got {_input_summary(bounds)}."
        )
    limit = _limit(max_facets, 1, "max_facets")
    facets = np.empty((limit, 3), dtype=np.int32)
    incident = np.empty((limit, 2), dtype=np.int32)
    endpoints = np.empty((limit, 2, 3), dtype=np.float64)
    kinds = np.empty((limit,), dtype=np.int32)
    items = np.empty((limit,), dtype=np.int32)
    count = np.zeros((1,), dtype=np.int64)
    counters = np.zeros((4,), dtype=np.int64)
    status = load_meshcore()["phx_mc_restricted_dual_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        cells.shape[0],
        cells.ctypes.data,
        bounds.ctypes.data,
        limit,
        _limit(work_limit, 1, "work_limit"),
        facets.ctypes.data,
        incident.ctypes.data,
        endpoints.ctypes.data,
        kinds.ctypes.data,
        items.ctypes.data,
        count.ctypes.data,
        counters.ctypes.data,
    )
    _raise_for_call(status, "restricted_dual_3d")
    if not 0 <= count[0] <= limit:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.restricted_dual_3d.points must satisfy the input contract (invalid restricted dual size); got {_input_summary(points)}.",
        )
    n = count[0]
    return RestrictedDual3D(
        facets=facets[:n],
        incident=incident[:n],
        endpoints=endpoints[:n],
        kinds=kinds[:n],
        item_status=items[:n],
        counters=RestrictedDualCounters.from_slots(counters),
    )


def restricted_centers_3d(
    points: ArrayLike,
    tetrahedra: ArrayLike,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Outward circumcenter intervals; failed rows retain per-cell status."""
    coordinates = _points(points, "points", 3)
    cells = _power_indices(tetrahedra, "tetrahedra")
    if (
        cells.ndim != 2
        or cells.shape[1] != 4
        or np.any(cells < 0)
        or np.any(cells >= coordinates.shape[0])
    ):
        raise ValueError(
            f"_meshcore.restricted_centers_3d.tetrahedra must contain valid four-vertex tuples; got {_input_summary(tetrahedra)}."
        )
    bounds = np.empty((cells.shape[0], 2, 3), dtype=np.float64)
    items = np.empty((cells.shape[0],), dtype=np.int32)
    status = load_meshcore()["phx_mc_restricted_centers_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        cells.shape[0],
        cells.ctypes.data,
        bounds.ctypes.data,
        items.ctypes.data,
    )
    _raise_for_call(status, "restricted_centers_3d")
    return bounds, items


def mixed_refinement_templates(
    cell_kind: str,
    /,
    *,
    axial: bool = False,
) -> tuple[NDArray[np.float64], ...]:
    """Validated same-family reference maps, identity followed by refinements."""
    match cell_kind:
        case "tetrahedron":
            kind = 0
        case "hexahedron":
            kind = 1
        case "prism":
            kind = 2
        case "pyramid":
            kind = 3
        case "quadrilateral":
            kind = 4
        case _:
            raise ValueError(
                f"_meshcore.mixed_refinement_templates.cell_kind must satisfy the input contract (Mixed templates require a quadrilateral, tetrahedron, hexahedron, prism or pyramid.); got {_input_summary(cell_kind)}."
            )
    if not isinstance(axial, (bool, np.bool_)):
        raise TypeError(
            f"_meshcore.mixed_refinement_templates.axial must be boolean; got {_input_summary(axial)}."
        )
    library = load_meshcore()
    counts = np.zeros((3,), dtype=np.int32)
    _raise_for_call(
        library["phx_mc_mixed_template_counts"](kind, int(axial), 0, counts.ctypes.data),
        "mixed_template_counts",
    )
    if counts[0] < 1:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR,
            f"_meshcore.mixed_refinement_templates.cell_kind must satisfy the input contract (missing mixed templates); got {_input_summary(cell_kind)}.",
        )
    total = int(counts[0])
    outputs: list[NDArray[np.float64]] = []
    for index in range(total):
        _raise_for_call(
            library["phx_mc_mixed_template_counts"](
                kind, int(axial), index, counts.ctypes.data
            ),
            "mixed_template_counts",
        )
        if counts[1] < 1 or counts[2] < 1:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                f"_meshcore.mixed_refinement_templates.cell_kind must satisfy the input contract (invalid mixed-template sizes); got {_input_summary(cell_kind)}.",
            )
        shape = (int(counts[1]), int(counts[2]), 2 if kind == 4 else 3)
        budget = current_native_execution_scope()
        points = (
            np.empty(shape, dtype=np.float64)
            if budget is None
            else budget.allocate_host_array(shape, np.float64)
        )
        _raise_for_call(
            library["phx_mc_mixed_template"](
                kind, int(axial), index, points.size, points.ctypes.data
            ),
            "mixed_template",
        )
        outputs.append(points)
    return tuple(outputs)


def restricted_rays_3d(
    points: ArrayLike,
    tetrahedra: ArrayLike,
    center_bounds: ArrayLike,
    facets: ArrayLike,
    incident_cells: ArrayLike,
    domain: ArrayLike,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]:
    """Enclose hull dual rays using source-bound circumcenter intervals."""
    coordinates = _points(points, "points", 3)
    tets = _power_indices(tetrahedra, "tetrahedra")
    faces = _power_indices(facets, "facets")
    cells = _power_indices(incident_cells, "incident_cells")
    centers = _real_array(center_bounds, "center_bounds")
    box = _real_array(domain, "domain")
    if (
        tets.ndim != 2
        or tets.shape[1] != 4
        or centers.shape != (tets.shape[0], 2, 3)
        or faces.ndim != 2
        or faces.shape[1] != 3
        or cells.shape != (faces.shape[0],)
        or np.any(tets < 0)
        or np.any(tets >= coordinates.shape[0])
        or np.any(faces < 0)
        or np.any(faces >= coordinates.shape[0])
        or np.any(cells < 0)
        or np.any(cells >= tets.shape[0])
        or box.shape not in ((6,), (2, 3))
    ):
        raise ValueError(
            f"_meshcore.restricted_rays_3d.facets must align with tetrahedra; got {_input_summary(facets)}."
        )
    box = np.ascontiguousarray(box.reshape((6,)), dtype=np.float64)
    if not np.all(np.isfinite(box)) or np.any(box[3:] <= box[:3]):
        raise ValueError(
            f"restricted_rays_3d.domain must be finite and ordered; got {_input_summary(box)}."
        )
    output = np.empty((faces.shape[0], 2, 2, 3), dtype=np.float64)
    kinds = np.empty((faces.shape[0],), dtype=np.int32)
    items = np.empty((faces.shape[0],), dtype=np.int32)
    status = load_meshcore()["phx_mc_restricted_rays_3d"](
        coordinates.shape[0],
        coordinates.ctypes.data,
        tets.shape[0],
        tets.ctypes.data,
        centers.ctypes.data,
        faces.shape[0],
        faces.ctypes.data,
        cells.ctypes.data,
        box.ctypes.data,
        output.ctypes.data,
        kinds.ctypes.data,
        items.ctypes.data,
    )
    _raise_for_call(status, "restricted_rays_3d")
    return output, kinds, items


__all__ = [
    "ArrangementCounters",
    "BoxHalfplaneClip",
    "BoxHalfspaceClip",
    "ConstrainedDelaunay2D",
    "ContactIntersections",
    "GraphPartition",
    "LevelSetCounters",
    "LevelSetOutputMode",
    "LevelSetRun",
    "LevelSetSplitRequest",
    "NativeMeshArrays",
    "NativeSourceComplex",
    "PeriodicDelaunay",
    "PolygonIntersectionSimplices",
    "RestrictedDual3D",
    "RestrictedDualCounters",
    "RestrictedPowerCounters",
    "SurfaceEnvelope",
    "SurfaceEnvelopeCounts",
    "SurfaceReconnection",
    "TetrahedronIntersectionSimplices",
    "level_set_split_3d",
    "current_native_execution_scope",
    "NativeExecutionLimits",
    "NativeExecutionScope",
    "NativeParallelExecution",
    "RemainingAllowance",
    "NativeBatchTicket",
    "is_native_host_buffer",
    "NativeExecutionEvidence",
    "NativePhaseRecorder",
    "NativeResourceFailure",
    "PlanarExecutionEvidence",
    "meshcore_runtime_identity",
    "subdivide_hex_grid",
    "IncrementalDelaunay3D",
    "IntersectionClass",
    "NativeArrangement",
    "ArrangementClassification",
    "ARRANGEMENT_UNCLASSIFIED",
    "ArrangementFailure",
    "MeshcoreError",
    "MeshcoreLibrary",
    "MeshcoreStatus",
    "Plc3DCounters",
    "PLC_3D_ENTITY_KINDS",
    "PLC_3D_FAILURE_REASONS",
    "PlcRecovery3D",
    "PlcRecoveryFailure",
    "PlcSourceConstraints",
    "TetMeshBoundaryPolicy",
    "TetMeshExudeCounters",
    "TetMeshImproveCounters",
    "TetMeshRefineCounters",
    "TET_MESH_UNMET_CRITERIA",
    "TET_MESH_UNMET_REASONS",
    "TetMesh3D",
    "TetMeshArrays",
    "TetMeshCurvedClaims",
    "TetMeshCurvedCollapsePlan",
    "TetMeshCurvedSources",
    "TetMeshInsertionPlan",
    "TetMeshEdgeConstruction",
    "TetMeshQuality",
    "TetMeshRun",
    "TetMeshSourceComplex",
    "TetMeshSourceEvidence",
    "TetMeshUnmet",
    "RestrictedPowerCells",
    "RestrictedPowerFailure",
    "RestrictedPowerOutputLimits",
    "RestrictedPowerPreflight",
    "RestrictedPowerCapacity",
    "GraphPartitionCounters",
    "PeriodicDelaunayEvidence",
    "SurfaceReconnectionCounters",
    "Triangulation3DStatistics",
    "mixed_refinement_templates",
    "restricted_centers_3d",
    "restricted_rays_3d",
    "clip_box_halfplanes",
    "clip_reference_triangle_exact",
    "ExactReferenceTetrahedronClip",
    "clip_reference_tetrahedron_exact",
    "ExactTetrahedronArrangement",
    "arrange_tetrahedra_exact",
    "polyhedron_corner_hexes",
    "arrange_triangles",
    "clip_box_halfspaces",
    "constrained_delaunay_2d",
    "delaunay_2d",
    "delaunay_3d",
    "exact_incircle",
    "exact_insphere",
    "exact_orient2d",
    "exact_orient3d",
    "graph_partition",
    "load_meshcore",
    "meshcore_available",
    "meshcore_exact_domain",
    "meshcore_identity",
    "meshcore_provider_installed",
    "periodic_delaunay",
    "point_triangle_locations",
    "polygon_intersection_moments",
    "polygon_intersection_simplices",
    "polyhedron_clip_moments",
    "regular_2d",
    "regular_3d",
    "recover_plc_3d",
    "plc_source_constraints",
    "restricted_power_cells",
    "restricted_dual_3d",
    "segment_triangle_intersections",
    "surface_envelope",
    "triangle_dual_quads",
    "tetrahedron_dual_hexes",
    "symbolic_incircle",
    "symbolic_insphere",
    "symbolic_orient2d",
    "interval_program_execute",
    "IntervalProgram",
    "simplex_quadratic_bounds",
    "symbolic_orient3d",
    "surface_reconnect",
    "tetrahedron_intersection_moments",
    "tetrahedron_intersection_simplices",
    "triangle_intersection_classes",
    "triangle_intersections",
]
