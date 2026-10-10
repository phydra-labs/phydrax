#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host binding of the native Phydrax meshcore library.

The optional ``phydrax-meshcore`` distribution ships a C++ shared library with a
plain C ABI (``native/meshcore/include/phydrax_meshcore.h``) that is loaded with
``ctypes``.  It owns exact geometric predicates (Shewchuk adaptive expansion
arithmetic with index-ordered symbolic perturbation), exactly classified convex
clipping moments, exact contact classification of triangles and segments in
3D, exact-predicate Delaunay/regular/constrained
triangulations, and the deterministic multilevel graph partition backend of
:mod:`phydrax.graph`.  Every function here is host-only NumPy interchange: inputs are
validated and converted to contiguous float64/int32 arrays, results are fresh
arrays, and native statuses reach the caller (per-item status arrays for batched
clipping, exceptions for rejected calls).

Library lookup order: the ``PHYDRAX_MESHCORE_LIBRARY`` environment variable
(path to the shared library), then ``phydrax_meshcore.library_path()``.  A
library is usable only when its exported ``phx_mc_abi_contract`` equals this
binding's C ABI contract (verified before any other entry point is bound), it
exports every C ABI symbol bound here, and its version equals the installed
``phydrax`` release; anything else is reported as
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
import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, is_dataclass, replace
from enum import Enum, IntEnum
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, final, Literal, overload, TYPE_CHECKING, TypeAlias

import numpy as np
from jax import Array, Device
from jax.extend.core import Jaxpr
from jax.sharding import Mesh
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import NDArray


if TYPE_CHECKING:
    from .meshing._measurements import NativeExecutionRecord
    from .meshing.providers._native_layer import PreparedLayerCore

_ENVIRONMENT = "PHYDRAX_MESHCORE_LIBRARY"
# SHA-256 of ``native/meshcore/include/phydrax_meshcore.h`` with CRLF normalized
# to LF: the public C declarations that ``_SIGNATURES`` binds. The native build
# exports the same digest of the header it compiled.
_ABI_CONTRACT = "01d7a8cf3845090029cee92843ace7a13813df40f4b96e1b3724ff62ef80d208"
_MAX_POINTS = 2147483646


class MeshcoreUnavailableError(ImportError):
    """The native meshcore library is not installed or cannot be loaded."""


class MeshcoreError(RuntimeError):
    """The native meshcore library refused a call for resource or internal reasons."""

    cut_failure_prefix: tuple[Any, ...] | None

    def __init__(
        self,
        status: MeshcoreStatus,
        message: str,
        /,
        *,
        work_evidence: NDArray[np.uint64] | None = None,
        memory_evidence: NDArray[np.uint64] | None = None,
    ) -> None:
        super().__init__(f"{message} (meshcore status {status.name})")
        self.status = status
        self.work_evidence = work_evidence
        self.memory_evidence = memory_evidence
        self.cut_failure_prefix = None


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


@final
@dataclass(frozen=True, slots=True)
class NativeExecutionEvidence:
    """Measured CPU-native phase work and managed allocations, including refusal."""

    work_evidence: NDArray[np.uint64]
    memory_evidence: NDArray[np.uint64]
    elapsed_seconds: float
    status: MeshcoreStatus
    externally_charged_work: int
    externally_charged_geometry_queries: int
    native_primitive_queries: int
    host_storage_live_bytes_upper: int
    host_storage_peak_bytes_upper: int
    prior_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for values in (self.work_evidence, self.memory_evidence):
            if values.dtype != np.dtype(np.uint64) or values.shape != (6,):
                raise ValueError("Native phase evidence requires six uint64 counters.")
            values.setflags(write=False)
        if (
            self.host_storage_live_bytes_upper < 0
            or self.host_storage_peak_bytes_upper < self.host_storage_live_bytes_upper
        ):
            raise ValueError(
                "Host storage evidence requires ordered nonnegative conservative byte bounds."
            )
        if (
            not math.isfinite(self.prior_elapsed_seconds)
            or self.prior_elapsed_seconds < 0.0
        ):
            raise ValueError("Imported elapsed evidence must be finite and nonnegative.")


@final
@dataclass(frozen=True, slots=True)
class NativeExecutionAllowance:
    """Actual nested remaining admission; observation never renews a budget."""

    remaining_work_units: int
    remaining_geometry_queries: int
    maximum_cavity_cells: int
    remaining_scratch_bytes: int
    remaining_wall_seconds: float
    status: MeshcoreStatus


def _free_host_array(library: MeshcoreLibrary, handle: int, /) -> None:
    library["phx_mc_execution_free_host_array"](handle)


@final
class NativeExecutionBudget:
    """One original CPU-native allowance, shared by nested FFI owners.

    This is not a launch or sharding runtime. The steady-clock deadline remains
    native throughout a monolithic call. ``mesh`` borrows an existing managed
    pool; scratch bounds additional live bytes above that pool's actual retained
    baseline. New construction borrows the active parent pool automatically.
    The scope's evidence survives success, resource refusal and Python errors.
    """

    def __init__(
        self,
        *,
        max_work: int,
        max_geometry_queries: int,
        max_cavity_cells: int,
        max_scratch_bytes: int,
        max_wall_seconds: float,
        mesh: TetMesh3D | None = None,
    ) -> None:
        if any(
            value is None
            for value in (
                max_work,
                max_geometry_queries,
                max_cavity_cells,
                max_scratch_bytes,
            )
        ):
            raise TypeError("Native execution limits must be explicit integers.")
        self._limits = tuple(
            _unsigned_budget(value, name)
            for value, name in (
                (max_work, "max_work"),
                (max_geometry_queries, "max_geometry_queries"),
                (max_cavity_cells, "max_cavity_cells"),
                (max_scratch_bytes, "max_scratch_bytes"),
            )
        )
        if isinstance(max_wall_seconds, (bool, np.bool_)) or not isinstance(
            max_wall_seconds, (int, float, np.integer, np.floating)
        ):
            raise TypeError("max_wall_seconds must be a nonnegative number.")
        self._seconds = float(max_wall_seconds)
        if math.isnan(self._seconds) or self._seconds < 0.0:
            raise ValueError("max_wall_seconds must be nonnegative.")
        if mesh is not None and not isinstance(mesh, TetMesh3D):
            raise TypeError("mesh must be a TetMesh3D.")
        self._mesh = mesh
        self._handle = ctypes.c_void_p()
        self._thread: int | None = None
        self._library: MeshcoreLibrary | None = None
        self.evidence: NativeExecutionEvidence | None = None
        self._token: Token[NativeExecutionBudget | None] | None = None
        self._parent: NativeExecutionBudget | None = None
        self._surfaced_child_statuses: set[MeshcoreStatus] = set()
        self._imported_preparation_receipts: dict[int, NativeExecutionRecord] = {}
        self._prepared_envelope_sources: set[str] = set()
        self._live_preparations: dict[int, PreparedLayerCore] = {}
        self._deferred_lock = threading.RLock()
        self._deferred_threads: set[int] = set()
        self._deferred_charges: list[tuple[int, int]] = []
        self._deferred_allowance: NativeExecutionAllowance | None = None

    @property
    def parent(self) -> NativeExecutionBudget | None:
        """The actual live ambient parent saved when this scope began."""
        previous = None if self._token is None else self._token.old_value
        return previous if isinstance(previous, NativeExecutionBudget) else None

    @property
    def deferred_worker_active(self) -> bool:
        """Whether this thread owns only deferred host-preparation admission."""
        return threading.get_ident() in self._deferred_threads

    def prepare_deferred_workers(self, /) -> None:
        """Snapshot admission for parallel immutable host preparation."""
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError(
                "Deferred workers require their active creating-thread budget."
            )
        with self._deferred_lock:
            if self._deferred_threads or self._deferred_charges:
                raise RuntimeError("Deferred worker preparation is already active.")
            self._deferred_allowance = self.remaining()

    @contextmanager
    def deferred_workers(self, /) -> Iterator[None]:
        """Flush all worker accounting after success or propagated failure.

        Worker executors must be entered inside this scope so their shutdown
        completes before the creating thread admits the accumulated charges.
        """
        self.prepare_deferred_workers()
        try:
            yield
        finally:
            self.flush_deferred_work()

    @contextmanager
    def deferred_worker(self, /) -> Iterator[None]:
        """Record host-only worker charges for creating-thread admission."""
        identity = threading.get_ident()
        with self._deferred_lock:
            if self._deferred_allowance is None or identity == self._thread:
                raise RuntimeError("Deferred worker preparation was not initialized.")
            self._deferred_threads.add(identity)
        token = _active_execution_budget.set(self)
        try:
            yield
        finally:
            _active_execution_budget.reset(token)
            with self._deferred_lock:
                self._deferred_threads.remove(identity)

    def flush_deferred_work(self, /) -> None:
        """Atomically admit all completed worker charges before publication."""
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError("Deferred work must flush on its creating thread.")
        with self._deferred_lock:
            if self._deferred_threads:
                raise RuntimeError("Deferred workers are still active.")
            work = sum(value[0] for value in self._deferred_charges)
            queries = sum(value[1] for value in self._deferred_charges)
            self._deferred_charges.clear()
            self._deferred_allowance = None
        self.charge(work=work, geometry_queries=queries)

    def __enter__(self) -> NativeExecutionBudget:
        if self._thread is not None or self.evidence is not None:
            raise RuntimeError("A native execution budget is single-use.")
        self._parent = current_native_execution_budget()
        library = load_meshcore()
        _raise_for_call(
            library["phx_mc_execution_begin"](
                *self._limits,
                self._seconds,
                None if self._mesh is None else self._mesh._live(),
                ctypes.addressof(self._handle),
            ),
            "execution_begin",
        )
        self._library = library
        self._thread = threading.get_ident()
        self._token = _active_execution_budget.set(self)
        return self

    def charge(self, *, work: int = 0, geometry_queries: int = 0) -> None:
        """Admit separately owned host/device batch work before dispatch.

        Do not repeat native owner work or predicates already charged by the
        scope. This single batch barrier is not a per-candidate host callback.
        """
        if work is None or geometry_queries is None:
            raise TypeError(
                "Actual batch work and query counts must be explicit integers."
            )
        identity = threading.get_ident()
        if identity != self._thread:
            with self._deferred_lock:
                if identity not in self._deferred_threads:
                    raise RuntimeError(
                        "Native batch admission requires the active creating thread."
                    )
                self._deferred_charges.append(
                    (
                        _unsigned_budget(work, "work"),
                        _unsigned_budget(geometry_queries, "geometry_queries"),
                    )
                )
            return
        if self._library is None:
            raise RuntimeError("Native batch admission requires an active budget.")
        _raise_for_call(
            self._library["phx_mc_execution_charge"](
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
                "Actual predecessor duration must be finite and nonnegative."
            )
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError("Preparation import requires the active creating thread.")
        _raise_for_call(
            self._library["phx_mc_execution_import_preparation"](
                self._handle,
                _unsigned_budget(work, "work"),
                _unsigned_budget(geometry_queries, "geometry_queries"),
                float(elapsed_seconds),
            ),
            "execution_import_preparation",
        )

    def admit_work_bound(self, maximum_work: int, /) -> None:
        """Dry-check a static pure-batch work bound without counting it as actual.

        Charge the batch's actual measured work after execution and before
        committing its candidates. No native owner delta is charged again.
        """
        if maximum_work is None:
            raise TypeError("maximum_work must be an explicit integer.")
        identity = threading.get_ident()
        if identity != self._thread:
            with self._deferred_lock:
                if (
                    identity not in self._deferred_threads
                    or self._deferred_allowance is None
                ):
                    raise RuntimeError(
                        "Native batch admission requires the active creating thread."
                    )
                if maximum_work > self._deferred_allowance.remaining_work_units:
                    raise MeshcoreError(
                        MeshcoreStatus.CAPACITY_EXCEEDED,
                        "execution_admit_work_bound: native work budget exceeded",
                    )
            return
        if self._library is None:
            raise RuntimeError("Native batch admission requires an active budget.")
        _raise_for_call(
            self._library["phx_mc_execution_admit_work_bound"](
                self._handle,
                _unsigned_budget(maximum_work, "maximum_work"),
            ),
            "execution_admit_work_bound",
        )

    def remaining(self) -> NativeExecutionAllowance:
        """Observe original parent minima and actually available managed bytes."""
        identity = threading.get_ident()
        if identity != self._thread:
            with self._deferred_lock:
                if (
                    identity not in self._deferred_threads
                    or self._deferred_allowance is None
                ):
                    raise RuntimeError(
                        "Native allowance observation requires its active creating thread."
                    )
                return self._deferred_allowance
        if self._library is None:
            raise RuntimeError("Native allowance observation requires an active budget.")
        values = np.zeros((4,), dtype=np.uint64)
        wall = np.zeros((1,), dtype=np.float64)
        status = MeshcoreStatus(
            self._library["phx_mc_execution_remaining"](
                self._handle,
                values.ctypes.data,
                wall.ctypes.data,
            )
        )
        if status == MeshcoreStatus.INVALID_ARGUMENT:
            _raise_for_call(status, "execution_remaining")
        return NativeExecutionAllowance(
            int(values[0]),
            int(values[1]),
            int(values[2]),
            int(values[3]),
            float(wall[0]),
            status,
        )

    def admit_cavity(self, count: int, /) -> None:
        """Admit an actual host edit's touched-cell set in the original ledger."""
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError(
                "Native cavity admission requires its active creating thread."
            )
        _raise_for_call(
            self._library["phx_mc_execution_admit_cavity"](
                self._handle,
                _unsigned_budget(count, "cavity_cells"),
            ),
            "execution_admit_cavity",
        )

    def host_workspace(self) -> NativeHostStorageWorkspace:
        """Reserve real Python-owned storage against this pool's original cap.

        Payloads remain in their actual Python owners. Their conservative bound
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

        Only this allocation's exact native metadata and payload requests are
        counted, in the same managed pool as native scratch. NumPy's Python
        array/buffer metadata and unrelated NumPy/JAX/compiler allocations are
        not measured. The buffer retains its native owner after scope exit;
        NumPy views keep it alive until the last buffer reference is released.
        """
        if (
            self._thread != threading.get_ident()
            or self._library is None
            or _active_execution_budget.get() is not self
        ):
            raise RuntimeError(
                "Native host allocation requires the active creating thread."
            )
        if not isinstance(shape, tuple) or any(
            isinstance(size, (bool, np.bool_)) or not isinstance(size, int)
            for size in shape
        ):
            raise TypeError("shape must be a tuple of nonnegative Python integers.")
        if len(shape) > 64:
            raise ValueError("shape exceeds NumPy's 64-dimensional array limit.")
        address_limit = np.iinfo(np.intp).max
        if any(size < 0 or size > address_limit for size in shape):
            raise ValueError("shape dimensions must be nonnegative and addressable.")
        if dtype is None:
            raise TypeError("dtype must be explicit.")
        element_dtype = np.dtype(dtype)
        if element_dtype.hasobject:
            raise TypeError("Native host arrays cannot contain object dtype.")
        if element_dtype.subdtype is not None:
            raise ValueError(
                "A top-level subarray dtype would change the requested shape."
            )
        # Empty dimensions still need representable C strides; the actual
        # payload request below remains exactly zero for an empty array.
        if math.prod(max(1, size) for size in shape) > (
            address_limit // max(1, element_dtype.itemsize)
        ):
            raise ValueError("shape and dtype exceed addressable NumPy array storage.")
        byte_count = math.prod(shape) * element_dtype.itemsize
        owner = ctypes.c_void_p()
        data = ctypes.c_void_p()
        library = self._library
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
        if owner.value is None or data.value is None:
            library["phx_mc_execution_free_host_array"](owner)
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "execution_allocate_host_array: missing buffer",
            )
        try:
            buffer = (ctypes.c_ubyte * byte_count).from_address(data.value)
            finalizer = weakref.finalize(buffer, _free_host_array, library, owner.value)
        except BaseException:
            _free_host_array(library, owner.value)
            raise
        try:
            # Interpreter atexit callbacks may still hold and use array views;
            # only actual buffer collection may release their native storage.
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
        if self._thread != threading.get_ident() or self._library is None:
            raise RuntimeError(
                "A native execution budget must end on its creating thread."
            )
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
            _raise_for_call(status, "execution_end")
        self._handle = ctypes.c_void_p()
        self._live_preparations.clear()
        self._thread = None
        if self._token is not None:
            _active_execution_budget.reset(self._token)
            self._token = None
        if status == MeshcoreStatus.OK and isinstance(error, MeshcoreError):
            status = error.status
        work = counters[:6]
        self.evidence = NativeExecutionEvidence(
            work,
            memory[:6],
            float(seconds[0]),
            status,
            int(counters[6]),
            int(counters[7]),
            int(counters[8]),
            int(memory[6]),
            int(memory[7]),
            float(prior_seconds[0]),
        )
        if isinstance(error, MeshcoreError):
            error.work_evidence = work
            error.memory_evidence = memory[:6]
            if status is error.status and self._parent is not None:
                # A nested scope shares the original native pool, whose terminal
                # status remains visible when the parent later ends. The child
                # exception is the consumer boundary for this refusal; retain
                # the parent's non-OK evidence without raising it a second time.
                self._parent._surfaced_child_statuses.add(status)
        if (
            error is None
            and status != MeshcoreStatus.OK
            and status not in self._surfaced_child_statuses
        ):
            raise MeshcoreError(
                status,
                "native execution allowance exhausted",
                work_evidence=work,
                memory_evidence=memory[:6],
            )


@final
class NativeHostStorageWorkspace:
    """A real external-host upper bound in the active native allocation pool.

    One opaque token is resized atomically: refusals preserve its old bound.
    Exit releases its bound even after a failed native scope. No dummy payload
    is allocated, and the native allocator's six measured counters stay separate.
    """

    def __init__(self, budget: NativeExecutionBudget, /) -> None:
        self._budget = budget
        self._library: MeshcoreLibrary | None = None
        self._reservation = ctypes.c_void_p()
        self._finalizer: weakref.finalize | None = None
        self._active = False
        self._token: Token[NativeHostStorageWorkspace | None] | None = None
        self.bound = 0
        self._retained_owners: dict[int, object] = {}
        self._logical_unmanaged_bytes_upper = 0

    def __enter__(self) -> NativeHostStorageWorkspace:
        if self._active or self._library is not None:
            raise RuntimeError("A native host-storage workspace is single-use.")
        if (
            current_native_execution_budget() is not self._budget
            or self._budget._library is None
        ):
            raise RuntimeError(
                "Host storage admission requires the active creating scope."
            )
        self._library = self._budget._library
        self._active = True
        self._token = _active_host_workspace.set(self)
        try:
            self.set_bound(0)
        except BaseException:
            self.close()
            raise
        return self

    def set_bound(self, total_bytes_upper: int, /) -> None:
        total = _unsigned_budget(total_bytes_upper, "total_bytes_upper")
        budget = current_native_execution_budget()
        if (
            not self._active
            or self._library is None
            or budget is None
            or budget._library is not self._library
        ):
            raise RuntimeError(
                "Host storage resizing requires its actual active native pool."
            )
        if total == 0 and self._reservation.value is None:
            return
        if total == self.bound and self._reservation.value is not None:
            return
        _raise_for_call(
            self._library["phx_mc_execution_reserve_host_storage"](
                budget._handle,
                total,
                ctypes.addressof(self._reservation),
            ),
            "execution_reserve_host_storage",
        )
        if self._reservation.value is None:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "Host storage admission lost its reservation.",
            )
        if self._finalizer is None:
            self._finalizer = weakref.finalize(
                self,
                self._library["phx_mc_execution_release_host_storage"],
                self._reservation.value,
            )
            self._finalizer.atexit = False
        self.bound = total

    @property
    def logical_unmanaged_bytes_upper(self) -> int:
        """Declared bytes of retained JAX arrays, not device/compiler measurement.

        These bytes participate in conservative admission through ``bound``.
        Their real device buffers and compiler allocations remain unmanaged.
        """
        return self._logical_unmanaged_bytes_upper

    def retain_owner(self, value: object, /) -> None:
        """Admit one real source tree and keep its distinct owners alive.

        Full dataclass fields are visited, including static numerical banks.
        NumPy views retain and charge their actual base owner once. JAX arrays
        contribute a conservative declared logical-bank upper bound, not a
        measured or enforced device/compiler allocation. Native-backed payloads
        already managed by the allocator are not counted again as host bytes.
        Mesh topology admission covers its Python wrapper, axis names and the
        actual Device pointer-bank/base owners; Device admission covers Python
        wrapper metadata only. Backend internals, compiler allocations and
        device buffers are not measured or claimed by this host upper bound.
        """
        from ._model._structure import _model_owner_children
        from .exterior._form_type import FormType, FormValueSpec

        if not self._active:
            raise RuntimeError("Source retention requires its active host workspace.")
        budget = current_native_execution_budget()
        if budget is None or budget._library is not self._library:
            raise RuntimeError("Source retention requires its actual active native pool.")
        starting_bound = self.bound
        starting_logical_unmanaged = self._logical_unmanaged_bytes_upper
        added: list[int] = []
        visiting: dict[int, object] = {}
        pending: list[tuple[object, bool, bool]] = []
        temporary_bound = 0

        def schedule(
            owner: object, leaving: bool = False, *, device_bank: bool = False
        ) -> None:
            nonlocal temporary_bound
            if not leaving and id(owner) in self._retained_owners:
                return
            budget.charge(work=1)
            self.set_bound(self.bound + 256)
            temporary_bound += 256
            pending.append((owner, leaving, device_bank))

        schedule(value)
        try:
            while pending:
                owner, leaving, device_bank = pending.pop()
                identity = id(owner)
                if leaving:
                    visiting.pop(identity)
                    self._retained_owners[identity] = owner
                    added.append(identity)
                    continue
                if identity in visiting:
                    raise ValueError("Source retention refuses cyclic owning graphs.")
                if (
                    isinstance(owner, np.ndarray)
                    and owner.dtype.hasobject
                    and not device_bank
                ):
                    raise TypeError(
                        "Source retention refuses object-dtype arrays outside Mesh device topology."
                    )
                if identity in self._retained_owners:
                    continue
                if isinstance(owner, Mesh):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    namespace = getattr(owner, "__dict__", None)
                    if isinstance(namespace, dict):
                        self.set_bound(self.bound + sys.getsizeof(namespace))
                    visiting[identity] = owner
                    schedule(owner, True)
                    schedule(owner.axis_names)
                    schedule(owner.devices, device_bank=True)
                    continue
                elif isinstance(owner, Device):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    namespace = getattr(owner, "__dict__", None)
                    if isinstance(namespace, dict):
                        visiting[identity] = owner
                        schedule(owner, True)
                        schedule(namespace)
                        continue
                elif isinstance(owner, Jaxpr):
                    # Current public JAX exposes open and closed forms as one
                    # immutable graph type whose ``jaxpr`` property is itself.
                    # Compiler equations are static metadata; actual constants
                    # remain independent numerical owners.
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    if owner.consts:
                        visiting[identity] = owner
                        schedule(owner, True)
                        schedule(owner.consts)
                        continue
                elif isinstance(owner, (Array, np.ndarray)):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    if isinstance(owner, Array):
                        self.set_bound(self.bound + owner.nbytes)
                        self._logical_unmanaged_bytes_upper += owner.nbytes
                    elif owner.dtype.hasobject:
                        # Only the actual Mesh devices bank and its real base
                        # owners enter this branch. Admit each visit before
                        # inspecting/scheduling its immutable Device wrapper.
                        for device in owner.flat:
                            budget.charge(work=1)
                            if not isinstance(device, Device):
                                raise TypeError(
                                    "Mesh topology banks require actual JAX Device elements."
                                )
                            schedule(device)
                        if owner.base is not None:
                            if (
                                not isinstance(owner.base, np.ndarray)
                                or not owner.base.dtype.hasobject
                            ):
                                raise TypeError(
                                    "Mesh topology banks require typed Device array base owners."
                                )
                            schedule(owner.base, device_bank=True)
                    elif owner.base is not None:
                        schedule(owner.base)
                elif isinstance(owner, memoryview):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    schedule(owner.obj)
                elif isinstance(owner, ctypes.Array):
                    finalizer = getattr(owner, "_native_execution_finalizer", None)
                    native = (
                        None
                        if not isinstance(finalizer, weakref.finalize)
                        else finalizer.peek()
                    )
                    managed = (
                        native is not None
                        and native[0] is owner
                        and native[1] is _free_host_array
                    )
                    self.set_bound(
                        self.bound
                        + sys.getsizeof(owner)
                        + 128
                        + (0 if managed else ctypes.sizeof(owner))
                    )
                elif (
                    owner is None
                    or owner is Ellipsis
                    or isinstance(
                        owner,
                        (
                            str,
                            bytes,
                            bytearray,
                            bool,
                            int,
                            float,
                            complex,
                            np.generic,
                            np.dtype,
                            Fraction,
                            Path,
                            slice,
                            type,
                            Enum,
                        ),
                    )
                ):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    if isinstance(owner, np.generic) and owner.dtype.hasobject:
                        raise TypeError("Source retention refuses object-dtype scalars.")
                    if isinstance(owner, Fraction):
                        schedule(owner.numerator)
                        schedule(owner.denominator)
                    elif isinstance(owner, slice):
                        schedule(owner.start)
                        schedule(owner.stop)
                        schedule(owner.step)
                    elif isinstance(owner, Enum):
                        schedule(owner.value)
                        schedule(owner.name)
                elif (
                    type(owner) is FormType
                    or type(owner) is FormValueSpec
                    or is_dataclass(owner)
                    or isinstance(owner, (Mapping, tuple, list, set, frozenset))
                ):
                    self.set_bound(self.bound + sys.getsizeof(owner) + 128)
                    namespace = getattr(owner, "__dict__", None)
                    if isinstance(namespace, dict):
                        self.set_bound(self.bound + sys.getsizeof(namespace))
                    visiting[identity] = owner
                    schedule(owner, True)
                    if type(owner) is FormType:
                        schedule(owner.dimension)
                        schedule(owner.degree)
                        schedule(owner.twist)
                        schedule(owner.fiber_shape)
                        schedule(owner.ambient_dimension)
                        schedule(owner.form_type_id)
                    elif type(owner) is FormValueSpec:
                        schedule(owner.form_type)
                        schedule(owner.proxy)
                        schedule(owner.value_spec_id)
                    else:
                        for child in _model_owner_children(owner):
                            schedule(child)
                    continue
                else:
                    raise TypeError(
                        f"Source retention refuses unsupported owner {type(owner).__qualname__}."
                    )
                self._retained_owners[identity] = owner
                added.append(identity)
            self.set_bound(self.bound - temporary_bound)
        except BaseException:
            self._logical_unmanaged_bytes_upper = starting_logical_unmanaged
            for identity in added:
                self._retained_owners.pop(identity)
            try:
                self.set_bound(starting_bound)
            except MeshcoreError:
                # The original failed native scope owns final resource status
                # and releases its token on exit; never replace that refusal.
                pass
            raise

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        elif self._library is not None and self._reservation.value is not None:
            self._library["phx_mc_execution_release_host_storage"](self._reservation)
        self._reservation = ctypes.c_void_p()
        self._active = False
        self._retained_owners.clear()
        self.bound = 0
        self._logical_unmanaged_bytes_upper = 0
        if self._token is not None:
            _active_host_workspace.reset(self._token)
            self._token = None

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


_active_execution_budget: ContextVar[NativeExecutionBudget | None] = ContextVar(
    "native_execution_budget",
    default=None,
)
_active_host_workspace: ContextVar[NativeHostStorageWorkspace | None] = ContextVar(
    "native_host_storage_workspace",
    default=None,
)


def current_native_host_workspace() -> NativeHostStorageWorkspace | None:
    """The innermost real host-storage owner of the active native pool."""
    workspace = _active_host_workspace.get()
    budget = current_native_execution_budget()
    if workspace is None or not workspace._active or budget is None:
        return None
    if workspace._library is not budget._library:
        raise RuntimeError(
            "A host workspace cannot move to another native allocation pool."
        )
    return workspace


def current_native_execution_budget() -> NativeExecutionBudget | None:
    """Return the actual ambient CPU-native owner without loading a library.

    A copied context cannot transfer a live native thread-local scope to another
    thread. Original request/operation-clock metadata remains Python-wrapper
    owned; this accessor exposes only the live native admission owner.
    """
    budget = _active_execution_budget.get()
    if budget is not None and (
        budget._thread != threading.get_ident()
        and threading.get_ident() not in budget._deferred_threads
    ):
        raise RuntimeError("A native execution budget belongs to its creating thread.")
    return budget


def charge_native_geometry_queries(count: int, /, *, work_units: int = 0) -> None:
    """Admit an owning source-evaluation batch before host/JAX execution.

    With no CPU-native phase this performs no library lookup or FFI call; the
    independent source owner's own declared limits remain authoritative.
    """
    if count is None or work_units is None:
        raise TypeError("Source work and query counts must be explicit integers.")
    budget = current_native_execution_budget()
    if budget is not None:
        budget.charge(work=work_units, geometry_queries=count)
    else:
        _unsigned_budget(count, "count")
        _unsigned_budget(work_units, "work_units")


@final
@dataclass(frozen=True, slots=True)
class PlanarExecutionEvidence:
    """Actual native measurements, including refused executions.

    ``work_evidence`` names work units, orientation/incircle/power requests,
    insertions, flips, peak cavity cells, work refusals and cavity refusals.
    ``memory_evidence`` names the byte limit, live/peak/last-request bytes,
    allocation count and refusal count. Both arrays are readonly uint64.
    """

    work_evidence: NDArray[np.uint64]
    memory_evidence: NDArray[np.uint64]
    status: MeshcoreStatus

    def __post_init__(self) -> None:
        if not isinstance(self.status, MeshcoreStatus):
            raise TypeError("status must be a MeshcoreStatus.")
        for array, size in ((self.work_evidence, 9), (self.memory_evidence, 6)):
            if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.uint64):
                raise TypeError("native execution evidence must use uint64 arrays.")
            if array.shape != (size,):
                raise ValueError(f"native execution evidence must have shape ({size},).")
            array.setflags(write=False)


@final
class NativeResourceFailure(MeshcoreError):
    """Native refusal preserving the actual planar execution measurements."""

    def __init__(self, evidence: PlanarExecutionEvidence, operation: str, /) -> None:
        super().__init__(
            evidence.status,
            f"{operation}: native execution refused",
            work_evidence=evidence.work_evidence,
            memory_evidence=evidence.memory_evidence,
        )


class IntersectionClass(IntEnum):
    """Exact contact class of a triangle pair or segment/triangle pair.

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

_SIGNATURES: dict[str, tuple[object, tuple[object, ...]]] = {
    "phx_mc_version": (ctypes.c_char_p, ()),
    "phx_mc_abi_contract": (ctypes.c_char_p, ()),
    "phx_mc_build_hash": (ctypes.c_char_p, ()),
    "phx_mc_build_configuration": (ctypes.c_char_p, ()),
    "phx_mc_exact_domain": (None, (_P, _P)),
    "phx_mc_execution_begin": (_I32, (_U64, _U64, _U64, _U64, _F64, _P, _P)),
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
        (_P, _I64, _P, _P, _I64, _I64, _I32, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_clip_box_halfspaces": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _I32,
            _P,
            _P,
            _P,
            _I32,
            _I32,
            _I32,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
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
    "phx_mc_constrained_delaunay_2d": (
        _I32,
        (
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I32,
            _F64,
            _F64,
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_mesh_dimension": (_I32, (_P,)),
    "phx_mc_mesh_point_count": (_I64, (_P,)),
    "phx_mc_mesh_cell_count": (_I64, (_P,)),
    "phx_mc_mesh_input_point_count": (_I64, (_P,)),
    "phx_mc_mesh_copy_points": (None, (_P, _P)),
    "phx_mc_mesh_copy_cells": (None, (_P, _P)),
    "phx_mc_mesh_copy_vertex_map": (None, (_P, _P)),
    "phx_mc_mesh_copy_cell_constraints": (None, (_P, _P)),
    "phx_mc_mesh_copy_cell_regions": (None, (_P, _P)),
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
    "phx_mc_surface_reconnect": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I32,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_periodic_delaunay": (
        _I32,
        (_I32, _I64, _P, _P, _P, _P, _F64, _I64, _I64, _P, _P),
    ),
    "phx_mc_periodic_cell_count": (_I64, (_P,)),
    "phx_mc_periodic_copy_cells": (None, (_P, _P, _P)),
    "phx_mc_periodic_free": (None, (_P,)),
    "phx_mc_plc3d_recover": (
        _I32,
        (
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _I32,
            _P,
            _P,
            _I64,
            _I64,
            _I64,
            _U64,
            _I32,
            _P,
        ),
    ),
    "phx_mc_plc3d_source_constraints": (
        _I32,
        (_I64, _P, _I64, _P, _P, _P, _I64, _P, _I64, _P, _I64, _U64, _I32, _P),
    ),
    "phx_mc_plc3d_sizes": (None, (_P, _P)),
    "phx_mc_plc3d_counters": (None, (_P, _P)),
    "phx_mc_plc3d_failure": (None, (_P, _P)),
    "phx_mc_plc3d_export": (None, (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_plc3d_source_witnesses": (None, (_P, _P, _P, _P, _P, _P)),
    "phx_mc_plc3d_free": (None, (_P,)),
    "phx_mc_plc3d_phase_times": (_I32, (_P, _P, _P, _P)),
    "phx_mc_plc3d_memory_evidence": (_I32, (_P, _P, _P)),
    "phx_mc_tet_mesh_create": (
        _I32,
        (
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I32,
            _I64,
            _I64,
            _I64,
            _P,
        ),
    ),
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
        (
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _I64,
            _I64,
            ctypes.c_int8,
            _P,
        ),
    ),
    "phx_mc_restricted_power_cells_exact": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _P,
            _I64,
            _I64,
            _I64,
            ctypes.c_int8,
            _P,
        ),
    ),
    "phx_mc_power_cells_sizes": (None, (_P, _P)),
    "phx_mc_power_cells_counters": (None, (_P, _P)),
    "phx_mc_power_cells_source_work_units": (_I64, (_P,)),
    "phx_mc_power_cells_failure": (None, (_P, _P)),
    "phx_mc_power_cells_export": (
        None,
        (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P),
    ),
    "phx_mc_power_cells_free": (None, (_P,)),
    "phx_mc_power_cells_export_vertex_carriers": (None, (_P, _P)),
    "phx_mc_power_cells_site_witness_size": (_I64, (_P,)),
    "phx_mc_power_cells_export_vertex_sites": (None, (_P, _P, _P)),
    "phx_mc_power_cells_phase_seconds": (None, (_P, _P)),
    "phx_mc_arrange_triangles": (
        _I32,
        (_I64, _P, _I64, _P, _P, _I64, _P, _I64, _I64, _P, _P),
    ),
    "phx_mc_arrangement_sizes": (None, (_P, _P)),
    "phx_mc_arrangement_counters": (None, (_P, _P)),
    "phx_mc_arrangement_export": (None, (_P, _P, _P, _P, _P, _P, _P, _P, _P, _P)),
    "phx_mc_arrangement_free": (None, (_P,)),
    "phx_mc_arrangement_classify": (_I32, (_P, _I64, _I64, _P, _P)),
    "phx_mc_arrangement_export_classification": (None, (_P, _P, _P)),
    "phx_mc_triangle_dual_quads": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_tetrahedron_dual_hexes": (_I32, (_I64, _I64, _P, _I64, _P)),
    "phx_mc_polyhedron_corner_hexes": (
        _I32,
        (
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_surface_envelope_create": (
        _I32,
        (
            _P,
            _I64,
            _P,
            _I64,
            _F64,
            _F64,
            _F64,
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
        ),
    ),
    "phx_mc_surface_envelope_arrays": (_I32, (_P, _P, _P, _P, _P)),
    "phx_mc_surface_envelope_free": (None, (_P,)),
    "phx_mc_level_set_split_3d": (
        _I32,
        (
            _I64,
            _P,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _I64,
            _P,
            _P,
            _I64,
            _I64,
            _I64,
            _I64,
            _P,
            _P,
            _P,
            _P,
            _P,
            _P,
        ),
    ),
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
    "phx_mc_tet_mesh_commit_edge_insertion": (
        _I32,
        (
            _P,
            _I32,
            _I32,
            _P,
            _P,
            _F64,
            _I64,
            _I32,
            _I64,
            _I64,
            _P,
            _I64,
            _P,
            _F64,
            _I64,
            _P,
        ),
    ),
    "phx_mc_tet_mesh_collapse_edge": (_I32, (_P, _I32, _I32, _I32, _I64, _P)),
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
    "phx_mc_tet_mesh_source_evidence": (_I32, (_P, _P, _P, _P, _P, _P, _P, _P)),
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
            return f"{_ENVIRONMENT} names a missing meshcore library: {configured!r}."
        return path
    if importlib.util.find_spec("phydrax_meshcore") is None:
        return (
            "The meshcore library is unavailable: install the optional "
            "'phydrax[meshcore]' extra or set "
            f"{_ENVIRONMENT} to libphydrax_meshcore."
        )
    return Path(importlib.import_module("phydrax_meshcore").library_path())


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
    return _resolve(os.environ.get(_ENVIRONMENT, "").strip())


def meshcore_available() -> bool:
    """Whether the native meshcore library can be loaded."""

    return not isinstance(_current(), str)


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
    """Actual loaded source/binary/configuration identity, excluding timings."""
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
        raise TypeError(f"{name} must be a real floating or integer array.")
    if np.issubdtype(array.dtype, np.integer) and np.any(
        (array > 2**53) | (array < -(2**53))
    ):
        raise ValueError(f"{name} integer values must be exactly representable.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _index_array(value: object, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be an integer array.")
    return array


def _points(value: object, name: str, width: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return array


def _batched_points(
    arrays: tuple[object, ...], names: tuple[str, ...], width: int, /
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    converted = tuple(
        _real_array(value, name) for value, name in zip(arrays, names, strict=True)
    )
    for array, name in zip(converted, names, strict=True):
        if array.ndim < 1 or array.shape[-1] != width:
            raise ValueError(f"{name} must have shape (..., {width}).")
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
            raise ValueError(f"{operation}: inputs must be finite.")
        case MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                f"{operation}: coordinates must be zero or have magnitude in "
                "[2**-120, 2**120] (weights [2**-240, 2**240]) for exact evaluation."
            )
        case MeshcoreStatus.DEGENERATE_INPUT:
            raise ValueError(f"{operation}: the distinct points do not span the space.")
        case MeshcoreStatus.INVALID_ARGUMENT | MeshcoreStatus.INVALID_INPUT:
            raise ValueError(f"{operation}: invalid input ({code.name}).")
        case MeshcoreStatus.CONSTRAINT_INTERSECTION:
            raise ValueError(
                f"{operation}: constraint segments cross in their interiors."
            )
        case MeshcoreStatus.CAPACITY_EXCEEDED:
            raise MeshcoreError(code, f"{operation}: explicit capacity exceeded")
        case MeshcoreStatus.TIMEOUT:
            raise MeshcoreError(code, f"{operation}: native wall-clock deadline reached")
        case MeshcoreStatus.REFINEMENT_LIMIT | MeshcoreStatus.INTERNAL_ERROR:
            raise MeshcoreError(code, f"{operation}: native failure")
        case _:
            raise ValueError(f"Unknown meshcore status {status}.")


# ---------------------------------------------------------------- predicates


def _predicate(
    name: str, arrays: tuple[object, ...], width: int, /, ids: object = None
) -> np.ndarray:
    library = load_meshcore()
    names = tuple(f"point {index}" for index in range(len(arrays)))
    flat, leading = _batched_points(arrays, names, width)
    count = flat[0].shape[0]
    signs = np.zeros((count,), dtype=np.int8)
    pointers = [array.ctypes.data for array in flat]
    if ids is None:
        status = library[name](count, *pointers, signs.ctypes.data)
    else:
        index = _index_array(ids, "ids")
        if index.shape[-1:] != (len(arrays),):
            raise ValueError(f"ids must have shape (..., {len(arrays)}).")
        index_flat = np.ascontiguousarray(
            np.broadcast_to(index, leading + (len(arrays),)).reshape((-1, len(arrays))),
            dtype=np.int64,
        )
        status = library[name](
            count, *pointers, index_flat.ctypes.data, signs.ctypes.data
        )
    _raise_for_call(status, name.removeprefix("phx_mc_"))
    return signs.reshape(leading)


def exact_orient2d(a: object, b: object, c: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a]`` for ``(..., 2)`` arrays (int8 in {-1, 0, 1})."""

    return _predicate("phx_mc_orient2d", (a, b, c), 2)


def exact_orient3d(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact ``sign det[b - a, c - a, d - a]`` for ``(..., 3)`` arrays."""

    return _predicate("phx_mc_orient3d", (a, b, c, d), 3)


def exact_incircle(a: object, b: object, c: object, d: object, /) -> np.ndarray:
    """Exact in-circle sign: positive when ``d`` is inside the circle of CCW ``(a, b, c)``."""

    return _predicate("phx_mc_incircle", (a, b, c, d), 2)


def exact_insphere(
    a: object, b: object, c: object, d: object, e: object, /
) -> np.ndarray:
    """Exact in-sphere sign: positive when ``e`` is inside the sphere of positive ``(a..d)``."""

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


# ---------------------------------------------------------------- clipping


def _counted_polygons(
    vertices: object, counts: object, name: str, /
) -> tuple[np.ndarray, np.ndarray]:
    array = _real_array(vertices, name)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError(f"{name} must have shape (n, k, 2).")
    count_array = _index_array(counts, f"{name} counts")
    if count_array.shape != array.shape[:1]:
        raise ValueError(f"{name} counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > array.shape[1]):
        raise ValueError(f"{name} counts must lie in [0, {array.shape[1]}].")
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
    orientation.  Returns ``(area (n,), first_moment (n, 2), status (n,))`` where
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
        raise ValueError("first and second polygon batches must have equal length.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convex polygon-pair intersections as triangle partitions plus their moments.

    Returns ``(simplices (n, S, 3, 2), simplex_counts (n,), area, first_moment,
    status)``.  Each nonempty intersection is fanned from the vertex average of
    its counterclockwise loop, the fan of ``polygon_intersection_moments``, so the
    signed triangle areas sum to ``area`` term by term.  The default capacity
    ``k1 + k2`` bounds every intersection.
    """

    library = load_meshcore()
    first, first_count = _counted_polygons(first_vertices, first_counts, "first_vertices")
    second, second_count = _counted_polygons(
        second_vertices, second_counts, "second_vertices"
    )
    if first.shape[0] != second.shape[0]:
        raise ValueError("first and second polygon batches must have equal length.")
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
    return simplices, simplex_counts, area, moment, status


def _tetrahedra(value: object, name: str, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (4, 3):
        raise ValueError(f"{name} must have shape (n, 4, 3).")
    return array


def _capacity(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1 or value > 2**31 - 1:
        raise ValueError(f"{name} must lie in [1, 2**31 - 1].")
    return int(value)


def tetrahedron_intersection_moments(
    first: object,
    second: object,
    /,
    *,
    vertex_capacity: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly classified intersection moments of tetrahedron pairs ``(n, 4, 3)``.

    Returns ``(volume (n,), first_moment (n, 3), status (n,))``.  The intersection
    has at most eight faces, so the default vertex capacity (48) is never binding
    for valid input.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tetrahedron-pair intersections as tetrahedron partitions plus their moments.

    Returns ``(simplices (n, S, 4, 3), simplex_counts (n,), volume, first_moment,
    status)``.  Each nonempty intersection is coned from the vertex average of its
    vertices over the fan triangles of every outward face, the fan of
    ``tetrahedron_intersection_moments``, so the signed simplex volumes sum to
    ``volume`` term by term.  The default capacity (20) bounds every intersection.
    """

    library = load_meshcore()
    first_array = _tetrahedra(first, "first")
    second_array = _tetrahedra(second, "second")
    if first_array.shape[0] != second_array.shape[0]:
        raise ValueError("first and second tetrahedron batches must have equal length.")
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
    return simplices, simplex_counts, volume, moment, status


def _halfspaces(
    normals: object, offsets: object, plane_counts: object, width: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_array = _real_array(normals, "normals")
    if normal_array.ndim != 3 or normal_array.shape[2] != width:
        raise ValueError(f"normals must have shape (n, m, {width}).")
    offset_array = _real_array(offsets, "offsets")
    if offset_array.shape != normal_array.shape[:2]:
        raise ValueError("offsets must have shape (n, m).")
    count_array = _index_array(plane_counts, "plane_counts")
    if count_array.shape != normal_array.shape[:1]:
        raise ValueError("plane_counts must have shape (n,).")
    if np.any(count_array < 0) or np.any(count_array > normal_array.shape[1]):
        raise ValueError(f"plane_counts must lie in [0, {normal_array.shape[1]}].")
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
    and ``tetrahedra`` ``(n, 4, 3)``.  Returns ``(volume, first_moment, status)``.
    The default vertex capacity ``4 (m + 4)`` bounds every intermediate polytope.
    """

    library = load_meshcore()
    normal_array, offset_array, count_array = _halfspaces(
        normals, offsets, plane_counts, 3
    )
    tetrahedron_array = _tetrahedra(tetrahedra, "tetrahedra")
    if tetrahedron_array.shape[0] != normal_array.shape[0]:
        raise ValueError("halfspace and tetrahedron batches must have equal length.")
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
        raise ValueError(f"box bounds must have shape ({width},).")
    if not np.all(lower_array < upper_array):
        raise ValueError("box_lower must be strictly below box_upper.")
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
) -> tuple[np.ndarray, ...]:
    """Clip one axis-aligned 2D box by per-item halfplane sets ``{x : n . x <= h}``.

    Returns ``(vertices (n, V, 2), edge_labels (n, V), vertex_counts (n,), area (n,),
    first_moment (n, 2), status (n,))``.  Edge ``k`` joins vertices ``k`` and
    ``k + 1`` (cyclic, counterclockwise) and is labeled with the item-local plane
    index or ``-(1 + 2 axis + side)`` for box sides.  The default capacity
    ``m + 4`` is exact for convex polygons.
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
    return vertices, labels, vertex_counts, area, moment, status


def clip_reference_triangle_exact(
    planes: tuple[tuple[Fraction, Fraction, Fraction], ...],
    /,
    *,
    maximum_scratch_bytes: int = 64 * 1024**2,
    maximum_work_units: int = 100_000,
) -> tuple[NDArray[np.int32], NDArray[np.int32], int]:
    """Native exact rational halfplanes; retain original boundary support pairs.

    Positive row denominator clearing and signed integer limb packing are exact.
    Native arbitrary-integer side tests decide every clipping transition. No
    rounded coefficient or vertex participates in topology. Source rows are
    (-1,0,0), (0,-1,0), (1,1,1), followed by three original target halfplanes.
    """
    if len(planes) != 6 or any(
        len(row) != 3 or any(not isinstance(value, Fraction) for value in row)
        for row in planes
    ):
        raise TypeError(
            "Exact reference clipping requires six three-Fraction halfplanes."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 < value <= np.iinfo(np.int64).max
        for value in (maximum_scratch_bytes, maximum_work_units)
    ):
        raise ValueError(
            "Exact reference clipping needs positive int64 scratch/work limits."
        )
    # The packed ABI payload and result storage are actual allocations in the
    # owning native pool. Python Fraction/int temporaries are not measured by
    # that pool and remain the caller's separate host-coefficient budget.
    with NativeExecutionBudget(
        max_work=maximum_work_units,
        max_geometry_queries=0,
        max_cavity_cells=0,
        max_scratch_bytes=maximum_scratch_bytes,
        max_wall_seconds=math.inf,
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
                "Exact reference integer clipping exceeds its scratch bound before packing.",
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
        # and returns its actual arithmetic work even on partial refusal.
        # Admit that local cap first, then charge the consumed work exactly
        # once before publishing any result. Packing visits are clock-only,
        # not a new CPU-instruction proxy for the original work quantity.
        scratch_limit = min(maximum_scratch_bytes, allowance.remaining_scratch_bytes)
        if scratch_limit < resident_bound:
            raise MeshcoreError(
                MeshcoreStatus.CAPACITY_EXCEEDED,
                "Exact reference integer clipping exceeds the remaining native scratch bound.",
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
@dataclass(frozen=True, slots=True)
class ExactReferenceTetrahedronClip:
    supporting_planes: NDArray[np.int32]
    incident_planes: NDArray[np.uint16]
    face_offsets: NDArray[np.int32]
    face_planes: NDArray[np.int32]
    face_vertices: NDArray[np.int32]
    work_units: int


def clip_reference_tetrahedron_exact(
    planes: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...],
    /,
    *,
    maximum_scratch_bytes: int,
    maximum_work_units: int,
) -> ExactReferenceTetrahedronClip:
    """Clip an original tetra reference chart with exact box-pullback planes.

    The first four planes are positive multiples of (-1,0,0,0),
    (0,-1,0,0), (0,0,-1,0), (1,1,1,-1). Six further original rational
    halfspaces use a*x+b*y+c*z+d<=0. Output is scientific plane incidence,
    never a rounded numerical replacement for original chart coefficients.
    """
    from contextlib import ExitStack

    if len(planes) != 10 or any(
        len(row) != 4 or any(not isinstance(value, Fraction) for value in row)
        for row in planes
    ):
        raise TypeError(
            "Exact tetra clipping requires ten four-Fraction original planes."
        )
    maximum_scratch_bytes = _limit(maximum_scratch_bytes, 1, "maximum_scratch_bytes")
    maximum_work_units = _limit(maximum_work_units, 1, "maximum_work_units")
    input_bits = max(
        sum(value.denominator.bit_length() for value in row)
        + max(value.numerator.bit_length() for value in row)
        for row in planes
    )
    temporary_bytes = 64 * (256 + 8 * ((input_bits + 29) // 30))
    if temporary_bytes > maximum_scratch_bytes:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            "Original exact tetra coefficients exceed host packing scratch before allocation.",
        )
    with (
        NativeExecutionBudget(
            max_work=maximum_work_units,
            max_geometry_queries=1,
            max_cavity_cells=0,
            max_scratch_bytes=maximum_scratch_bytes,
            max_wall_seconds=math.inf,
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
            resident = 32768 + max(1, max(sizes)) * 65536
            offsets = budget.allocate_host_array((41,), np.int64)
            signs = budget.allocate_host_array((40,), np.int8)
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
            supports = budget.allocate_host_array((16, 3), np.int32)
            incidence = budget.allocate_host_array((16,), np.uint16)
            face_offsets = budget.allocate_host_array((11,), np.int32)
            face_offsets[0] = 0
            face_planes = budget.allocate_host_array((10,), np.int32)
            face_vertices = budget.allocate_host_array((48,), np.int32)
            counts = budget.allocate_host_array((2,), np.int32)
            work = budget.allocate_host_array((1,), np.int64)
            counts.fill(0)
            work.fill(0)
            allowance = budget.remaining()
            _raise_for_call(
                allowance.status, "clip_reference_tetrahedron_exact_allowance"
            )
            scratch = min(maximum_scratch_bytes, allowance.remaining_scratch_bytes)
            if scratch < resident:
                raise MeshcoreError(
                    MeshcoreStatus.CAPACITY_EXCEEDED,
                    "Original exact tetra predicates exceed remaining native scratch.",
                )
            work_limit = min(maximum_work_units, allowance.remaining_work_units)
            budget.admit_work_bound(max(1, work_limit))
            library = load_meshcore()
            call = library["phx_mc_clip_reference_tetrahedron_exact"](
                packed.ctypes.data,
                packed.size,
                offsets.ctypes.data,
                signs.ctypes.data,
                scratch,
                work_limit,
                16,
                10,
                48,
                supports.ctypes.data,
                incidence.ctypes.data,
                face_offsets.ctypes.data,
                face_planes.ctypes.data,
                face_vertices.ctypes.data,
                counts.ctypes.data,
                counts[1:].ctypes.data,
                work.ctypes.data,
            )
            # Native decoding/predicates charge the original scope themselves.
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
    """Native SCI corner-link templates over actual canonical cut incidence."""
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
                f"{name} must be an authoritative contiguous signed-64-bit incidence bank."
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
            "Cut corner incidence requires edge pairs and one-dimensional CSR banks."
        )
    if (
        not isinstance(corner_orientations, np.ndarray)
        or corner_orientations.dtype != np.dtype(np.int8)
        or not corner_orientations.flags.c_contiguous
        or corner_orientations.shape != cell_vertices.shape
    ):
        raise TypeError(
            "Exact source corner orientations must bind every original cell-vertex occurrence."
        )
    nv = _limit(original_vertex_count, 1, "original_vertex_count")
    maximum_cells = _limit(maximum_cells, 1, "maximum_cells")
    maximum_scratch_bytes = _limit(maximum_scratch_bytes, 1, "maximum_scratch_bytes")
    maximum_work_units = _limit(maximum_work_units, 1, "maximum_work_units")
    nf, nc = face_offsets.size - 1, cell_offsets.size - 1
    if nf <= 0 or nc <= 0 or vertex_offsets.size != nc + 1:
        raise ValueError(
            "Cut corner CSR must retain complete original source faces and cells."
        )
    count = cell_vertices.size
    if count > maximum_cells:
        raise MeshcoreError(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            "Actual cut corner templates exceed original cell capacity.",
        )
    with NativeExecutionBudget(
        max_work=maximum_work_units,
        max_geometry_queries=0,
        max_cavity_cells=0,
        max_scratch_bytes=maximum_scratch_bytes,
        max_wall_seconds=math.inf,
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
            nv,
            edges.shape[0],
            nf,
            nc,
            face_vertices.size,
            cell_faces.size,
            cell_vertices.size,
            edges.ctypes.data,
            face_offsets.ctypes.data,
            face_vertices.ctypes.data,
            cell_offsets.ctypes.data,
            cell_faces.ctypes.data,
            vertex_offsets.ctypes.data,
            cell_vertices.ctypes.data,
            corner_orientations.ctypes.data,
            nv + edges.shape[0] + nf + nc,
            maximum_cells,
            min(maximum_scratch_bytes, allowance.remaining_scratch_bytes),
            min(maximum_work_units, allowance.remaining_work_units),
            output.ctypes.data,
            parents.ctypes.data,
            completed.ctypes.data,
            witness.ctypes.data,
            work.ctypes.data,
        )
        if call == MeshcoreStatus.REFINEMENT_LIMIT:
            raise MeshcoreError(
                MeshcoreStatus.REFINEMENT_LIMIT,
                f"Original cut corner link (cell row, vertex row, edge degree, face degree) {tuple(int(value) for value in witness)} requires conforming grid-link regularization.",
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
) -> tuple[np.ndarray, ...]:
    """Clip one axis-aligned 3D box by per-item halfspace sets ``{x : n . x <= h}``.

    Returns ``(vertices (n, V, 3), vertex_counts (n,), face_offsets (n, F + 1),
    face_labels (n, F), face_vertices (n, W), face_counts (n,), volume (n,),
    first_moment (n, 3), status (n,))``.  Faces are sorted by label, oriented
    counterclockwise from outside, and start at their smallest vertex.  Default
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
        count,
        lower.ctypes.data,
        upper.ctypes.data,
        normal_array.shape[1],
        normal_array.ctypes.data,
        offset_array.ctypes.data,
        count_array.ctypes.data,
        vertex_capacity_,
        face_capacity_,
        face_vertex_capacity_,
        vertices.ctypes.data,
        vertex_counts.ctypes.data,
        face_offsets.ctypes.data,
        face_labels.ctypes.data,
        face_vertices.ctypes.data,
        face_counts.ctypes.data,
        volume.ctypes.data,
        moment.ctypes.data,
        status.ctypes.data,
    )
    _raise_for_call(call, "clip_box_halfspaces")
    return (
        vertices,
        vertex_counts,
        face_offsets,
        face_labels,
        face_vertices,
        face_counts,
        volume,
        moment,
        status,
    )


# ---------------------------------------------------------------- contacts in 3D


def _simplices(value: object, name: str, corners: int, /) -> np.ndarray:
    array = _real_array(value, name)
    if array.ndim != 3 or array.shape[1:] != (corners, 3):
        raise ValueError(f"{name} must have shape (n, {corners}, 3).")
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
        raise ValueError("vertex ids must be given for both inputs or for neither.")
    first = _index_array(first_ids, "first_ids")
    second = _index_array(second_ids, "second_ids")
    if first.shape != shapes[0] or second.shape != shapes[1]:
        raise ValueError(f"vertex ids must have shapes {shapes[0]} and {shapes[1]}.")
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
        raise ValueError("both input batches must have equal length.")
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
    """Exact :class:`IntersectionClass` of triangle pairs ``(n, 3, 3)``.

    Returns ``(classes (n,) int8, status (n,) int32)``.  Optional vertex ids
    ``(n, 3)`` for both inputs make adjacency identity-based: a vertex contact
    is shared only when the ids agree, so coincident positions with distinct
    ids are ``TOUCHING``; ids repeated within a triangle or equal across
    triangles at distinct positions report ``INVALID_INPUT``.  Without ids,
    coincident positions are shared.  Failed items (``NONFINITE_INPUT``,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact contact classes and bounded contact constructions of triangle pairs.

    Returns ``(classes (n,), point_counts (n,), points (n, 6, 3), point_bounds
    (n, 6), point_features (n, 6, 2), status (n,))`` with the classes, ids and
    statuses of :func:`triangle_intersection_classes`.  The intersection is
    empty, a point, a segment (endpoints ordered along the intersection line)
    or a convex polygon of at most six vertices, counterclockwise about the
    first triangle's normal.  ``point_features[i, k]`` holds the feature of the
    first and second triangle containing point ``k`` exactly: 0-2 vertex ``j``,
    3-5 the edge opposite vertex ``j - 3``, 6 the relative interior.  Points
    that are input vertices are exact (bound 0); the others are constructions
    ``p + t (q - p)`` with ``t`` a ratio of exact orientation expansions and
    ``point_bounds`` a rigorous max-norm distance bound from the exact point.
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
    return classes, counts, points, bounds, features, status


def segment_triangle_intersections(
    segments: object,
    triangles: object,
    /,
    *,
    segment_ids: object | None = None,
    triangle_ids: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact contact classes and constructions of segments ``(n, 2, 3)`` with triangles.

    Returns ``(classes (n,), point_counts (n,), points (n, 2, 3), point_bounds
    (n, 2), point_features (n, 2, 2), status (n,))`` as
    :func:`triangle_intersections`; segment features are 0 and 1 for the
    endpoints and 6 for the interior.  ``SHARED_EDGE`` means the segment is an
    edge of the triangle, ``COPLANAR_OVERLAP`` a coplanar contact through the
    triangle's interior.  A zero-length segment or collinear triangle reports
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
    return classes, counts, points, bounds, features, status


def point_triangle_locations(
    points: object, triangles: object, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact location of points ``(n, 3)`` relative to triangles ``(n, 3, 3)``.

    Returns ``(sides (n,) int8, features (n,) int8, status (n,) int32)``:
    ``sides`` is ``orient3d(t0, t1, t2, x)`` and ``features`` the triangle
    feature containing the point (0-2 vertex, 3-5 edge opposite vertex ``j - 3``,
    6 interior) or -1 when it is off the closed triangle.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 3)
    triangle_array = _simplices(triangles, "triangles", 3)
    if point_array.shape[0] != triangle_array.shape[0]:
        raise ValueError("points and triangles must have equal length.")
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
    library: MeshcoreLibrary, handle: ctypes.c_void_p, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The native handle is released even when a copy fails.
    try:
        dimension = library["phx_mc_mesh_dimension"](handle)
        point_count = library["phx_mc_mesh_point_count"](handle)
        cell_count = library["phx_mc_mesh_cell_count"](handle)
        input_count = library["phx_mc_mesh_input_point_count"](handle)
        points = np.zeros((point_count, dimension), dtype=np.float64)
        cells = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        vertex_map = np.zeros((input_count,), dtype=np.int32)
        cell_constraints = np.zeros((cell_count, dimension + 1), dtype=np.int32)
        cell_regions = np.zeros((cell_count,), dtype=np.int32)
        library["phx_mc_mesh_copy_points"](handle, points.ctypes.data)
        library["phx_mc_mesh_copy_cells"](handle, cells.ctypes.data)
        library["phx_mc_mesh_copy_vertex_map"](handle, vertex_map.ctypes.data)
        library["phx_mc_mesh_copy_cell_constraints"](handle, cell_constraints.ctypes.data)
        library["phx_mc_mesh_copy_cell_regions"](handle, cell_regions.ctypes.data)
    finally:
        library["phx_mc_mesh_free"](handle)
    return points, cells, vertex_map, cell_constraints, cell_regions


def _limit(value: int | None, default: int, name: str, /) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if not 1 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a positive signed-int64 value.")
    return int(value)


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
) -> tuple[np.ndarray, np.ndarray]:
    library = load_meshcore()
    point_array = _points(points, "points", width)
    count = point_array.shape[0]
    if count > _MAX_POINTS:
        raise ValueError("meshcore triangulations support fewer than 2**31 - 1 points.")
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
            raise ValueError("weights must have shape (n,).")
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
        PlanarExecutionEvidence(work, memory, MeshcoreStatus(status))
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
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, f"{operation}: missing mesh")
    _, cells, vertex_map, _, _ = _read_mesh(library, handle)
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
    """Exact Delaunay triangulation of ``(n, 2)`` points.

    Returns ``(triangles (T, 3) int32, vertex_map (n,) int32)``; triangles are
    counterclockwise, canonically ordered, and cocircular ties are resolved by
    index-ordered symbolic perturbation.  ``vertex_map`` sends duplicate points
    to their smallest identical index.

    All limits default to signed-int64 maximum; zero is a hard bound.
    ``record_native_resources`` receives actual work and allocation evidence,
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
    """Exact Delaunay tetrahedralization of ``(n, 3)`` points.

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
    """Exact regular (weighted Delaunay) triangulation; redundant points map to -1."""

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
    points: object, weights: object, /, *, max_tetrahedra: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exact regular (weighted Delaunay) tetrahedralization; redundant points map to -1."""

    return _triangulate("phx_mc_regular_3d", points, weights, 3, max_tetrahedra)


def constrained_delaunay_2d(
    points: object,
    segments: object,
    /,
    *,
    holes: object | None = None,
    keep_convex_hull: bool = False,
    min_angle: float = 0.0,
    max_area: float = float("inf"),
    max_steiner: int = 0,
    max_triangles: int | None = None,
    max_cavity_cells: int | None = None,
    max_work: int | None = None,
    max_scratch_bytes: int | None = None,
    record_native_resources: Callable[[PlanarExecutionEvidence], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MeshcoreStatus]:
    """Constrained Delaunay triangulation with Ruppert/Chew refinement.

    ``segments`` are ``(m, 2)`` point indices.  Without ``keep_convex_hull`` the
    triangles outside the segment-bounded region and inside hole seeds are
    removed.  ``min_angle`` (degrees, ``[0, 60)``), ``max_area`` and
    ``max_steiner`` bound the refinement. Returns ``(points (N, 2), triangles
    (T, 3), vertex_map (input_count,), segment_ids (T, 3), status)``.
    ``vertex_map`` explicitly maps authored input identities to output vertices;
    ``segment_ids[t, k]`` is the input segment carrying the edge opposite vertex
    ``k`` (or -1), and ``status`` is ``OK`` or ``REFINEMENT_LIMIT`` (valid mesh,
    quality targets unmet within ``max_steiner``).

    ``max_cavity_cells``, ``max_work`` and ``max_scratch_bytes`` apply before
    allocation and throughout preparation, insertion and refinement. All
    limits default to signed-int64 maximum; zero is a hard bound. The optional
    resource observer receives readonly :class:`PlanarExecutionEvidence`
    without changing the numerical tuple return.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 2)
    segment_array = _index_array(segments, "segments")
    if segment_array.ndim != 2 or segment_array.shape[1] != 2:
        raise ValueError("segments must have shape (m, 2).")
    if np.any(segment_array < 0) or np.any(segment_array >= point_array.shape[0]):
        raise ValueError("segments reference a missing point.")
    segment_int = np.ascontiguousarray(segment_array, dtype=np.int32)
    hole_array = (
        np.zeros((0, 2), dtype=np.float64)
        if holes is None
        else _points(holes, "holes", 2)
    )
    if not isinstance(keep_convex_hull, bool):
        raise TypeError("keep_convex_hull must be a bool.")
    angle = float(min_angle)
    area = float(max_area)
    if not (0.0 <= angle < 60.0):
        raise ValueError("min_angle must lie in [0, 60) degrees.")
    if not area > 0.0:
        raise ValueError("max_area must be positive (inf disables the bound).")
    if isinstance(max_steiner, bool) or not isinstance(max_steiner, (int, np.integer)):
        raise TypeError("max_steiner must be an integer.")
    if max_steiner < 0:
        raise ValueError("max_steiner must be nonnegative.")
    total = point_array.shape[0] + int(max_steiner)
    if total > _MAX_POINTS:
        raise ValueError("meshcore triangulations support fewer than 2**31 - 1 points.")
    limit = _budget(max_triangles, "max_triangles")
    work = np.empty((9,), dtype=np.uint64)
    memory = np.empty((6,), dtype=np.uint64)
    handle = ctypes.c_void_p()
    status = library["phx_mc_constrained_delaunay_2d"](
        point_array.shape[0],
        point_array.ctypes.data,
        segment_int.shape[0],
        segment_int.ctypes.data,
        hole_array.shape[0],
        hole_array.ctypes.data,
        1 if keep_convex_hull else 0,
        angle,
        area,
        int(max_steiner),
        limit,
        _budget(max_cavity_cells, "max_cavity_cells"),
        _budget(max_work, "max_work"),
        _budget(max_scratch_bytes, "max_scratch_bytes"),
        work.ctypes.data,
        memory.ctypes.data,
        ctypes.addressof(handle),
    )
    evidence = PlanarExecutionEvidence(work, memory, MeshcoreStatus(status))
    code = MeshcoreStatus(status)
    if code not in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT):
        if record_native_resources is not None:
            record_native_resources(evidence)
        if code in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.INTERNAL_ERROR):
            raise NativeResourceFailure(evidence, "constrained_delaunay_2d")
        _raise_for_call(status, "constrained_delaunay_2d")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "constrained_delaunay_2d: missing mesh"
        )
    mesh_points, cells, vertex_map, cell_segments, _ = _read_mesh(library, handle)
    if record_native_resources is not None:
        record_native_resources(evidence)
    return mesh_points, cells, vertex_map, cell_segments, code


TRIANGULATION_3D_STATISTICS = (
    "vertex_ids",
    "live_vertices",
    "finite_cells",
    "ghost_cells",
    "cell_slots",
    "free_slots",
    "constrained_facets",
    "work",
    "committed_insertions",
    "refused_insertions",
    "largest_cavity",
    "retained_bytes",
    "peak_retained_bytes",
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
    earlier identical point, or to -1 (a refused insertion).  The result depends
    only on the vertex set and ids, never on insertion order.  ``max_vertices``
    bounds the ids, ``max_tetrahedra`` the finite cells and ``max_cavity`` the
    tetrahedra one insertion may remove; every refusal is decided before any
    change, so refused insertions leave the triangulation unchanged.  The
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
            raise ValueError("max_vertices must lie in [n, 2**31 - 2].")
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
                MeshcoreStatus.INTERNAL_ERROR, "triangulation_3d_create: missing handle"
            )
        self._library = library
        self._handle = handle.value
        self._finalizer = weakref.finalize(
            self, _free_triangulation, library, handle.value
        )

    def _live(self) -> int:
        if not self._finalizer.alive:
            raise ValueError("the incremental triangulation is closed.")
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
        point on a constrained facet).  Non-finite or out-of-domain points, or
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
        repeated ids, or a different existing reference.  Later insertions never
        cross a constrained facet.
        """

        handle = self._live()
        facet_array = _index_array(facets, "facets")
        id_array = _index_array(constraint_ids, "constraint_ids")
        if facet_array.ndim != 2 or facet_array.shape[1] != 3:
            raise ValueError("facets must have shape (m, 3).")
        if id_array.shape != facet_array.shape[:1]:
            raise ValueError("constraint_ids must have shape (m,).")
        if np.any(np.abs(facet_array) > _MAX_POINTS) or np.any(
            np.abs(id_array) > 2**31 - 1
        ):
            raise ValueError("facet vertex ids and references must fit int32.")
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
        labeled differently.  Cells created later inherit the region of the
        cells they replace.
        """

        handle = self._live()
        seed_array = _points(seeds, "seeds", 3)
        label_array = _index_array(labels, "labels")
        if label_array.shape != seed_array.shape[:1]:
            raise ValueError("labels must have shape (s,).")
        if np.any(np.abs(label_array) > 2**31 - 1):
            raise ValueError("labels must fit int32.")
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
        """Exact location of ``(m, 3)`` points.

        Returns ``(cells (m, 4) int32, locations (m,) int8, status (m,) int32)``:
        the canonical vertex tuple (even permutation starting at its smallest
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

    def statistics(self) -> np.ndarray:
        """Int64 counters named by :data:`TRIANGULATION_3D_STATISTICS`."""

        handle = self._live()
        values = np.zeros((len(TRIANGULATION_3D_STATISTICS),), dtype=np.int64)
        call = self._library["phx_mc_triangulation_3d_statistics"](
            handle, values.ctypes.data
        )
        _raise_for_call(call, "triangulation_3d_statistics")
        return values

    def finalize(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Canonical snapshot; the triangulation stays usable.

        Returns ``(points (N, 3), tetrahedra (T, 4), vertex_map (N,),
        facet_constraints (T, 4), regions (T,))``: every submitted point, the
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
                MeshcoreStatus.INTERNAL_ERROR, "triangulation_3d_finalize: missing mesh"
            )
        return _read_mesh(self._library, mesh)


# ---------------------------------------------------------------- graph partition

GRAPH_PARTITION_COUNTERS = (
    "coarsening_levels",
    "coarsest_vertex_count",
    "matched_pairs",
    "bisection_trials",
    "refinement_passes",
    "moves_committed",
    "moves_rolled_back",
    "balance_moves",
    "nonempty_repairs",
    "adjacency_visits",
    "candidate_evaluations",
)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Multilevel k-way partition of a canonical symmetric weighted CSR graph.

    Returns ``(parts, counters)``: int32 part per vertex and the int64 work
    counters named by :data:`GRAPH_PARTITION_COUNTERS`. The semantic owner,
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
        raise ValueError("graph_partition: misaligned CSR or part arrays.")
    parts = np.zeros((count,), dtype=np.int32)
    counters = np.zeros((len(GRAPH_PARTITION_COUNTERS),), dtype=np.int64)
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
            f"graph_partition: work limit {work_limit} exhausted after "
            f"{counters[9]} adjacency visits and {counters[10]} candidate evaluations",
        )
    _raise_for_call(status, "graph_partition")
    return parts, counters


# ---------------------------------------------------------------- surface reconnection

SURFACE_RECONNECTION_COUNTERS = (
    "inserted",
    "refused",
    "flips",
    "walk_steps",
    "work",
    "exhausted",
)


def _surface_metric_buffer(
    value: object | None, count: int, name: str, /
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = _real_array(value, name)
    if array.shape != (count, 3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain one finite ambient three-dimensional tensor per vertex."
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
            f"{name} must contain signed-int64 pole IDs or -1 for ordinary vertices."
        )
    return np.ascontiguousarray(array, dtype=np.int64)


def surface_reconnect(
    charts: object,
    points: object,
    normals: object,
    triangles: object,
    constrained: object,
    insert_charts: object,
    insert_points: object,
    insert_normals: object,
    insert_spacing: object,
    insert_hints: object,
    /,
    *,
    sweep: bool,
    max_triangles: int,
    work_limit: int,
    vertex_metrics: object | None = None,
    insert_metrics: object | None = None,
    vertex_pole_ids: object | None = None,
    insert_pole_ids: object | None = None,
    insert_edges: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    ``sweep`` flips every edge afterwards. Returns ``(triangles, constrained,
    vertex_ids, item_status, counters)``: accepted points receive ids ``n,
    n + 1, ...`` (``-1`` when refused), ``item_status`` holds
    :class:`MeshcoreStatus` codes and ``counters`` the values named by
    :data:`SURFACE_RECONNECTION_COUNTERS`. An invalid chart triangulation raises
    ``ValueError``.
    ``insert_edges`` optionally selects existing unconstrained reciprocal edges
    by local source vertex rows, with shape ``(k, 2)``; ``(-1, -1)`` requests
    ordinary point insertion. A refused selected edge is never reinterpreted
    as a generic point insertion. With ``sweep=False``, a noncollinear rounded
    point is the execution view of a producer-owned exact rational edge
    restriction. A declared radial pole complex may defer physical-child
    validation to the producer, which must validate the exact chart and
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
            "surface_reconnect: misaligned vertex, cell or insertion arrays."
        )
    edges = None
    if insert_edges is not None:
        edge_values = np.asarray(insert_edges)
        if edge_values.shape != (count, 2):
            raise ValueError("insert_edges must have shape (insert_count, 2).")
        if edge_values.dtype.kind not in "iu":
            raise TypeError("insert_edges must contain integer vertex rows.")
        if np.any(edge_values < -1) or np.any(edge_values >= point_array.shape[0]):
            raise ValueError(
                "insert_edges vertex rows are outside the source vertex range."
            )
        generic = np.all(edge_values == -1, axis=1)
        if np.any(np.any(edge_values == -1, axis=1) & ~generic):
            raise ValueError("insert_edges generic rows must be paired (-1, -1).")
        if np.any((edge_values[:, 0] == edge_values[:, 1]) & ~generic):
            raise ValueError("insert_edges selected endpoints must be distinct.")
        if np.any(edge_values > np.iinfo(np.int32).max):
            raise ValueError("insert_edges vertex rows must fit int32.")
        if not np.all(generic):
            edges = np.ascontiguousarray(edge_values, dtype=np.int32)
    metrics = _surface_metric_buffer(
        vertex_metrics, point_array.shape[0], "vertex_metrics"
    )
    new_metrics = _surface_metric_buffer(insert_metrics, count, "insert_metrics")
    if metrics is not None and new_metrics is None and count == 0:
        new_metrics = np.empty((0, 3, 3), dtype=np.float64)
    if (metrics is None) != (new_metrics is None):
        raise ValueError("vertex_metrics and insert_metrics must be supplied together.")
    poles = _surface_pole_buffer(vertex_pole_ids, point_array.shape[0], "vertex_pole_ids")
    new_poles = _surface_pole_buffer(insert_pole_ids, count, "insert_pole_ids")
    capacity = _limit(max_triangles, cells.shape[0] + 2 * count, "max_triangles")
    triangles_out = np.zeros((capacity, 3), dtype=np.int32)
    flags_out = np.zeros((capacity, 3), dtype=np.int8)
    produced = np.zeros((1,), dtype=np.int64)
    vertex_ids = np.full((count,), -1, dtype=np.int32)
    item_status = np.zeros((count,), dtype=np.int32)
    counters = np.zeros((len(SURFACE_RECONNECTION_COUNTERS),), dtype=np.int64)
    status = library["phx_mc_surface_reconnect"](
        chart_array.shape[0],
        chart_array.ctypes.data,
        point_array.ctypes.data,
        normal_array.ctypes.data,
        _pointer(metrics),
        _pointer(poles),
        cells.shape[0],
        cells.ctypes.data,
        flags.ctypes.data,
        count,
        new_charts.ctypes.data,
        new_points.ctypes.data,
        new_normals.ctypes.data,
        _pointer(new_metrics),
        _pointer(new_poles),
        spacing.ctypes.data,
        hints.ctypes.data,
        _pointer(edges),
        _boolean_flag(sweep, "sweep"),
        capacity,
        _budget(work_limit, "work_limit"),
        triangles_out.ctypes.data,
        flags_out.ctypes.data,
        produced.ctypes.data,
        vertex_ids.ctypes.data,
        item_status.ctypes.data,
        counters.ctypes.data,
    )
    _raise_for_call(status, "surface_reconnect")
    rows = int(produced[0])
    return (
        triangles_out[:rows],
        flags_out[:rows].astype(np.bool_),
        vertex_ids,
        item_status,
        counters,
    )


PERIODIC_DELAUNAY_EVIDENCE = (
    "rounds",
    "margin",
    "image_count",
    "uncertified_cells",
    "required_margin",
    "finite_cell_slots",
    "exact_evaluations",
    "perturbed_decisions",
    "exhausted_limit",
    "duplicate_representative",
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
) -> tuple[MeshcoreStatus, np.ndarray, np.ndarray, np.ndarray]:
    """Periodic Delaunay triangulation on certified bounded image neighborhoods.

    ``points`` (n, d) are representatives with unwrapped lattice coordinates
    ``fractional`` (finite, magnitude at most 2**30); each representative's base
    image is wrapped into [0, 1) by the integer shift ``-floor(fractional)``.
    ``lattice`` (d, d) holds lattice vectors as rows and ``inverse`` (d, d) maps
    Cartesian to fractional coordinates. Returns
    ``(status, vertices (T, d + 1) int32, shifts (T, d + 1, d) int32, evidence)``
    with ``evidence`` named by :data:`PERIODIC_DELAUNAY_EVIDENCE`. The domain
    outcomes ``OK``, ``CAPACITY_EXCEEDED`` (image or cell budget, see
    ``exhausted_limit``) and ``INVALID_INPUT`` (a repeated orbit) are returned
    with their evidence for the semantic owner to report; every other status,
    including out-of-range fractional coordinates, raises.
    """

    library = load_meshcore()
    point_array = _real_array(points, "points")
    if point_array.ndim != 2 or point_array.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3).")
    count, dimension = point_array.shape
    fractional_array = _points(fractional, "fractional", dimension)
    lattice_array = _real_array(lattice, "lattice")
    inverse_array = _real_array(inverse, "inverse")
    if (
        fractional_array.shape != point_array.shape
        or lattice_array.shape != (dimension, dimension)
        or inverse_array.shape != (dimension, dimension)
    ):
        raise ValueError("periodic_delaunay: misaligned point or lattice arrays.")
    if count < 1 or count > _MAX_POINTS:
        raise ValueError("periodic_delaunay requires between 1 and 2**31 - 2 points.")
    evidence = np.zeros((len(PERIODIC_DELAUNAY_EVIDENCE),), dtype=np.float64)
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
            return (
                status,
                np.zeros((0, dimension + 1), dtype=np.int32),
                np.zeros((0, dimension + 1, dimension), dtype=np.int32),
                evidence,
            )
        case _:
            _raise_for_call(status, "periodic_delaunay")
    if not handle.value:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "periodic_delaunay: missing triangulation"
        )
    # The native handle is released even when a copy fails.
    try:
        cells = library["phx_mc_periodic_cell_count"](handle)
        vertices = np.zeros((cells, dimension + 1), dtype=np.int32)
        shifts = np.zeros((cells, dimension + 1, dimension), dtype=np.int32)
        library["phx_mc_periodic_copy_cells"](
            handle, vertices.ctypes.data, shifts.ctypes.data
        )
    finally:
        library["phx_mc_periodic_free"](handle)
    return status, vertices, shifts, evidence


# ---------------------------------------------------------------- PLC recovery

TetMeshBoundaryPolicy: TypeAlias = Literal["fixed", "conforming"]

PLC_3D_COUNTERS = (
    "rounds",
    "input_triangles",
    "facet_groups",
    "plc_edges",
    "protected_vertices",
    "segment_steiner_points",
    "facet_steiner_points",
    "cavities",
    "largest_cavity",
    "gift_wrap_candidates",
    "contact_tests",
    "work_units",
    "peak_retained_bytes",
    "interior_steiner_points",
)
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
    :data:`PLC_3D_COUNTERS`. ``source_refusal`` is the certified deviation
    and the declared bound of a ``"source_deviation"`` refusal (zeros
    otherwise).
    """

    def __init__(
        self,
        status: MeshcoreStatus,
        reason: str,
        entities: tuple[tuple[str, int], ...],
        counters: np.ndarray,
        /,
        *,
        source_refusal: tuple[float, float] = (0.0, 0.0),
        memory_evidence: NDArray[np.uint64] | None = None,
    ) -> None:
        described = ", ".join(f"{kind} {index}" for kind, index in entities)
        super().__init__(
            status,
            f"plc3d_recover: {reason} ({described})",
            memory_evidence=memory_evidence,
        )
        self.reason = reason
        self.entities = entities
        self.counters = counters
        self.source_refusal = source_refusal


@final
@dataclass(frozen=True, slots=True)
class PlcRecovery3D:
    """Recovered constrained tetrahedralization of a PLC (host arrays).

    ``points`` keep the input vertices first; Steiner points follow with
    ``vertex_dimension`` 1 (on a PLC edge), 2 (on a facet) or 3 (interior
    cavity-fill point). Domain
    tetrahedra are positively oriented with their region; ``faces`` are every
    constrained subfacet oriented like its source facet (``face_sources``);
    ``segments`` the subsegments of every PLC edge (``segment_sources`` index
    ``plc_edges``: explicit segments first, then facet-group boundary edges).
    ``protection_radii`` are the protecting-ball radii (0 unprotected);
    ``input_triangles`` are the exact triangulation of the input polygons
    (``input_polygons`` names each triangle's polygon). Every point carries a
    source witness: stratum (0 none, 1 PLC edge, 2 input triangle), the
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
    counters: np.ndarray
    memory_evidence: NDArray[np.uint64] | None


@dataclass(frozen=True, slots=True)
class PlcSourceConstraints:
    """Validated original PLC strata; no tetrahedral recovery is implied."""

    points: NDArray[np.float64]
    plc_edges: NDArray[np.int32]
    input_triangles: NDArray[np.int32]
    input_polygons: NDArray[np.int32]
    counters: NDArray[np.int64]
    memory_evidence: NDArray[np.uint64] | None

    def __post_init__(self) -> None:
        for values in (
            self.points,
            self.plc_edges,
            self.input_triangles,
            self.input_polygons,
            self.counters,
        ):
            values.setflags(write=False)
        if self.memory_evidence is not None:
            self.memory_evidence.setflags(write=False)


def _plc_result_evidence(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    status: int,
    record_native_phase: Callable[[str, float, int | None, int], None] | None,
    /,
) -> tuple[NDArray[np.int64], NDArray[np.uint64] | None]:
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
    memory_values.setflags(write=False)
    memory = memory_values if memory_enabled[0] else None
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
                MeshcoreStatus.INTERNAL_ERROR, "invalid PLC phase measurements"
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
    counters = np.zeros((len(PLC_3D_COUNTERS),), dtype=np.int64)
    library["phx_mc_plc3d_counters"](handle, counters.ctypes.data)
    if status != MeshcoreStatus.OK:
        failure = np.zeros((5,), dtype=np.int64)
        library["phx_mc_plc3d_failure"](handle, failure.ctypes.data)
        refusal = np.zeros((2,), dtype=np.float64)
        empty = np.empty((0,), dtype=np.float64)
        library["phx_mc_plc3d_source_witnesses"](
            handle,
            empty.ctypes.data,
            empty.ctypes.data,
            empty.ctypes.data,
            empty.ctypes.data,
            refusal.ctypes.data,
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
            memory_evidence=memory,
        )
    return counters, memory


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
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
) -> PlcSourceConstraints:
    """Prepare exact PLC edge/polygon strata without constructing a volume.

    Source ordering is identical to recovery's source preparation. Allocation,
    validation, and work refusals retain actual native evidence; optional phase
    measurements do not become fields of the returned scientific artifact.
    """
    library = load_meshcore()
    point_array = _points(points, "points", 3)
    offset_values = _index_array(polygon_offsets, "polygon_offsets")
    if np.any(offset_values < 0) or np.any(offset_values > np.iinfo(np.int64).max):
        raise ValueError("polygon_offsets must contain nonnegative signed-int64 values.")
    offsets = np.ascontiguousarray(offset_values, dtype=np.int64)
    loops = _power_indices(polygon_vertices, "polygon_vertices")
    facets = _power_indices(polygon_facets, "polygon_facets")
    regions = _power_indices(facet_regions, "facet_regions")
    edges = (
        np.zeros((0, 2), dtype=np.int32)
        if segments is None
        else _power_indices(segments, "segments")
    )
    if (
        offsets.ndim != 1
        or facets.ndim != 1
        or offsets.shape[0] != facets.shape[0] + 1
        or loops.ndim != 1
        or regions.ndim != 2
        or regions.shape[1] != 2
        or edges.ndim != 2
        or edges.shape[1] != 2
        or (offsets.size and offsets[-1] != loops.shape[0])
    ):
        raise ValueError("plc_source_constraints: misaligned PLC arrays.")
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
            "plc3d_source_constraints: missing result",
        )
    try:
        counters, memory = _plc_result_evidence(
            library, handle, status, record_native_phase
        )
        sizes = np.empty((6,), dtype=np.int64)
        library["phx_mc_plc3d_sizes"](handle, sizes.ctypes.data)
        count, cells, faces, pieces, table, triangles = sizes.tolist()
        if count != point_array.shape[0] or cells or faces or pieces:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR,
                "Source-only PLC preparation produced a volume.",
            )
        source_points = np.empty((count, 3), dtype=np.float64)
        source_edges = np.empty((table, 2), dtype=np.int32)
        source_triangles = np.empty((triangles, 3), dtype=np.int32)
        source_polygons = np.empty((triangles,), dtype=np.int32)
        empty = np.empty((0,), dtype=np.int32)
        protection = np.empty((count,), dtype=np.float64)
        dimensions = np.empty((count,), dtype=np.int8)
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
    points: object,
    polygon_offsets: object,
    polygon_vertices: object,
    polygon_facets: object,
    facet_regions: object,
    /,
    *,
    segments: object | None = None,
    seeds: object | None = None,
    seed_regions: object | None = None,
    facet_tolerances: ArrayLike | None = None,
    segment_tolerances: ArrayLike | None = None,
    boundary_policy: TetMeshBoundaryPolicy,
    max_vertices: int,
    max_tetrahedra: int,
    work_limit: int,
    max_scratch_bytes: int | None = None,
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
) -> PlcRecovery3D:
    """Validate an oriented PLC and recover it in a constrained tetrahedralization.

    ``facet_regions`` (F, 2) names the region on the positive side of each
    facet's right-hand loop normal and on its negative side (-1 void). A
    ``"fixed"`` boundary admits no boundary Steiner point; ``"conforming"``
    splits PLC edges and facets at exactly representable points on them or,
    under a positive declared ``facet_tolerances`` (F,) / ``segment_tolerances``
    (S,) bound (``None``: zero, exact only), at ancestry-backed carriers whose
    certified deviation that bound admits. A scientific refusal raises
    :class:`PlcRecoveryFailure` with its evidence; malformed arguments raise
    ``ValueError``.

    ``max_scratch_bytes=None`` selects ordinary untracked native allocation
    (UINT64_MAX); zero is a hard allocation cap. ``memory_evidence`` is present
    on success/failure only when an actual allocator owner supplied the six
    uint64 measurements, not an estimated size or fabricated zero ledger.
    """

    library = load_meshcore()
    point_array = _points(points, "points", 3)
    offset_values = _index_array(polygon_offsets, "polygon_offsets")
    if np.any(offset_values < 0) or np.any(offset_values > np.iinfo(np.int64).max):
        raise ValueError("polygon_offsets must contain nonnegative signed-int64 values.")
    offsets = np.ascontiguousarray(offset_values, dtype=np.int64)
    loops = _power_indices(polygon_vertices, "polygon_vertices")
    facets = _power_indices(polygon_facets, "polygon_facets")
    regions = _power_indices(facet_regions, "facet_regions")
    edges = (
        np.zeros((0, 2), dtype=np.int32)
        if segments is None
        else _power_indices(segments, "segments")
    )
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
            raise ValueError(f"{name} must be finite, nonnegative and one per entity.")
    if (
        offsets.ndim != 1
        or offsets.shape[0] != facets.shape[0] + 1
        or loops.ndim != 1
        or regions.ndim != 2
        or regions.shape[1] != 2
        or edges.ndim != 2
        or edges.shape[1] != 2
        or seed_labels.shape != (seed_points.shape[0],)
        or (offsets.size and offsets[-1] != loops.shape[0])
    ):
        raise ValueError("recover_plc_3d: misaligned PLC arrays.")
    match boundary_policy:
        case "fixed":
            policy = 0
        case "conforming":
            policy = 1
        case _:
            raise ValueError(f"Unknown boundary policy {boundary_policy!r}.")
    handle = ctypes.c_void_p()
    status = library["phx_mc_plc3d_recover"](
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
        seed_points.shape[0],
        seed_points.ctypes.data,
        seed_labels.ctypes.data,
        policy,
        None if facet_bounds is None else facet_bounds.ctypes.data,
        None if segment_bounds is None else segment_bounds.ctypes.data,
        _budget(max_vertices, "max_vertices"),
        _budget(max_tetrahedra, "max_tetrahedra"),
        _budget(work_limit, "work_limit"),
        _unsigned_budget(max_scratch_bytes, "max_scratch_bytes"),
        int(record_native_phase is not None),
        ctypes.addressof(handle),
    )
    if not handle.value:
        _raise_for_call(status, "plc3d_recover")
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "plc3d_recover: missing result"
        )
    # The native result is released even when a copy fails.
    try:
        counters, memory = _plc_result_evidence(
            library, handle, status, record_native_phase
        )
        sizes = np.zeros((6,), dtype=np.int64)
        library["phx_mc_plc3d_sizes"](handle, sizes.ctypes.data)
        count, cells, faces, pieces, table, triangles = (int(value) for value in sizes)
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
        )
        refusal = np.zeros((2,), dtype=np.float64)
        library["phx_mc_plc3d_source_witnesses"](
            handle,
            result.witness_strata.ctypes.data,
            result.witness_entities.ctypes.data,
            result.witness_parameters.ctypes.data,
            result.witness_deviations.ctypes.data,
            refusal.ctypes.data,
        )
    finally:
        library["phx_mc_plc3d_free"](handle)
    return result


# ------------------------------------------------ constrained tetrahedral mesh

TET_MESH_REFINE_COUNTERS = (
    "circumcenter_insertions",
    "subfacet_splits",
    "subsegment_splits",
    "encroachment_deferrals",
    "refused_insertions",
    "queue_pops",
    "unmet_cells",
    "work_units",
    "largest_cavity",
    "vertices",
)
TET_MESH_IMPROVE_COUNTERS = (
    "face_removals",
    "edge_removals",
    "relocations",
    "vertex_removals",
    "passes",
    "attempts",
    "remaining_slivers",
    "work_units",
    "vertex_insertions",
    "multiface_removals",
)
TET_MESH_EXUDE_COUNTERS = (
    "trials",
    "flips",
    "weighted_vertices",
    "passes",
    "remaining_slivers",
    "work_units",
)
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
@dataclass(frozen=True, slots=True)
class TetMeshRun:
    """Status and counters of one refinement or improvement run.

    ``status`` is ``OK`` (every criterion met), ``REFINEMENT_LIMIT`` (valid
    mesh; unmet cells listed by :meth:`TetMesh3D.unmet`) or
    ``CAPACITY_EXCEEDED`` (a work, vertex or cell limit stopped the run; the
    mesh is valid). ``counters`` follow :data:`TET_MESH_REFINE_COUNTERS`,
    :data:`TET_MESH_IMPROVE_COUNTERS` or :data:`TET_MESH_EXUDE_COUNTERS`.
    """

    status: MeshcoreStatus
    counters: np.ndarray


@final
@dataclass(frozen=True, slots=True)
class TetMeshQuality:
    """Shape distribution of the finite cells (angles in degrees).

    ``dihedral_histogram`` counts per-cell smallest dihedral angles in 5 degree
    bins over ``[0, 90)``; ``slivers`` counts cells below the requested angle.
    """

    minimum_dihedral: float
    maximum_dihedral: float
    maximum_radius_edge: float
    mean_radius_edge: float
    minimum_volume: float
    total_volume: float
    mean_minimum_dihedral: float
    dihedral_histogram: np.ndarray
    slivers: int


@final
@dataclass(frozen=True, slots=True)
class TetMeshUnmet:
    """Cells that violate a criterion after the last run and why.

    ``criteria`` index :data:`TET_MESH_UNMET_CRITERIA`, ``reasons``
    :data:`TET_MESH_UNMET_REASONS`; ``values`` are the measured radius-edge
    ratio, circumradius-to-size ratio, smallest dihedral angle (degrees), the
    shared floating orientation filter's positive-determinant lower bound, or
    for ``validity`` the exact ``det**2 / (floor**2 * prod |e_k|**2) - 1`` margin
    of the relative determinant floor (``e_k`` the edges from the cell's
    canonical first vertex; nonpositive below the floor).
    """

    tetrahedra: np.ndarray
    criteria: np.ndarray
    reasons: np.ndarray
    values: np.ndarray


@final
@dataclass(frozen=True, slots=True)
class TetMeshArrays:
    """Canonical snapshot of a :class:`TetMesh3D`.

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
@dataclass(frozen=True, slots=True, eq=False)
class TetMeshInsertionPlan:
    """Immutable inspected final conflict cavity tied to one accepted native state.

    The original source witness and finite cell tuples own no reconstructed
    mesh/classification. The native owner rechecks their generation and exact
    final geometry/source reconstruction before its one atomic commit.
    """

    _owner: TetMesh3D
    first: int
    second: int
    split_position: NDArray[np.float64]
    position: NDArray[np.float64]
    source_fraction: float
    maximum_cavity_cells: int
    insertion_kind: int
    accepted_generation: int
    removed: NDArray[np.int32]
    proposed: NDArray[np.int32]
    buffer_bytes: int


def _free_tet_mesh(library: MeshcoreLibrary, handle: int, /) -> None:
    library["phx_mc_tet_mesh_free"](handle)


def _rows(value: object, name: str, width: int, /) -> np.ndarray:
    array = _power_indices(value, name)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    return array


def _labels(value: object, name: str, count: int, /) -> np.ndarray:
    array = _power_indices(value, name)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},).")
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
            f"{name} must contain nonnegative signed-int64 scientific labels with shape ({count},)."
        )
    return np.ascontiguousarray(labels, dtype=np.int64)


def _source_group_rows(
    labels: NDArray[np.int64],
    scientific_ids: NDArray[np.int64],
    name: str,
    /,
) -> NDArray[np.int32]:
    """Lower actual IDs only through their explicit immutable scientific bank."""
    positions = np.searchsorted(scientific_ids, labels)
    if labels.size and (
        not scientific_ids.size
        or np.any(positions >= scientific_ids.size)
        or not np.array_equal(
            scientific_ids[np.minimum(positions, scientific_ids.size - 1)], labels
        )
    ):
        raise ValueError(
            f"{name} names a source absent from its declared scientific ID bank."
        )
    if scientific_ids.size > _MAX_POINTS:
        raise ValueError(
            "Native source group row count exceeds the signed-int32 indexing capacity."
        )
    return np.ascontiguousarray(positions, dtype=np.int32)


def _budget(value: int | None, name: str, /) -> int:
    if value is None:
        return 2**63 - 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a nonnegative signed-int64 value.")
    return int(value)


def _unsigned_budget(value: int | None, name: str, /) -> int:
    if value is None:
        return 2**64 - 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if not 0 <= value <= np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must be a nonnegative uint64 value.")
    return int(value)


def _node_id(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer vertex ID.")
    if not 0 <= value <= np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a nonnegative signed-int32 vertex ID.")
    return int(value)


def _boolean_flag(value: bool, name: str, /) -> int:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean.")
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
            raise MeshcoreError(code, f"{operation}: unexpected status")


@final
@dataclass(frozen=True, slots=True, eq=False)
class TetMeshEdgeConstruction:
    """Nonmutating native edge proposal and its original source-row witness."""

    position: NDArray[np.float64]
    parameter: float
    witness_stratum: int
    witness_entity: int
    witness_parameters: NDArray[np.float64]
    witness_deviation: float


@final
@dataclass(frozen=True, slots=True)
class TetMeshSourceComplex:
    """Declared source complex with an independent immutable source-point bank.

    ``faces`` (R, 3) and ``segments`` (Q, 2) index ``points``, with
    native row selectors ``face_group_rows``/``segment_group_rows`` into the
    canonical actual int64 scientific banks ``face_ids``/``segment_ids``.
    Deviation bounds ``face_tolerances``/``segment_tolerances`` are nonnegative
    (zero: exact); a group row is never a scientific identity.
    Per initial point, ``witness_strata`` (0 none, 1 segment row, 2 triangle
    row) with ``witness_entities`` and ``witness_parameters`` (N, 2): a line
    parameter from the row's first endpoint, or the weights of the triangle
    row's second and third corners. The native owner recertifies every
    witness deviation and every face/segment ancestry; nothing is trusted.
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
                    "Declared native PLC source banks require their canonical host dtypes."
                )
            if bank.ndim != rank:
                raise ValueError("Declared native PLC source banks have invalid ranks.")
        for groups, identifiers in (
            (self.face_group_rows, self.face_ids),
            (self.segment_group_rows, self.segment_ids),
        ):
            if np.any(identifiers < 0) or np.any(identifiers[1:] <= identifiers[:-1]):
                raise ValueError(
                    "Scientific source ID banks must be nonnegative, distinct and canonically ordered."
                )
            if np.any(groups < 0) or np.any(groups >= identifiers.size):
                raise ValueError(
                    "Native source group rows index undeclared scientific IDs."
                )
        _source_complex(self, self.witness_strata.shape[0])
        for bank, _, _ in banks:
            bank.setflags(write=False)


@final
@dataclass(frozen=True, slots=True)
class TetMeshSourceEvidence:
    """Source fidelity of a native tetrahedral mesh.

    ``requested_bound`` is the largest declared row bound and
    ``achieved_bound`` the largest certified deviation of a live vertex (zero
    exactly when every vertex lies on its source). ``face_ancestors`` and
    ``segment_ancestors`` name the source row of each exported face/segment
    (-1 none). Per vertex the witness stratum, row, parameters and certified
    deviation. ``refusals`` are the bounded carriers the latest refinement
    refused: represented edges (K, 2), stratum, source row, certified
    deviation and declared bound.
    """

    requested_bound: float
    achieved_bound: float
    face_ancestors: NDArray[np.int32]
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


def _source_complex(
    source: TetMeshSourceComplex, count: int, /
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.int8],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
]:
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
        raise ValueError("The declared source complex is misaligned with the carrier.")
    return (
        faces,
        face_ids,
        face_bounds,
        segments,
        segment_ids,
        segment_bounds,
        strata,
        entities,
        parameters,
        points,
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
    own exact source complex; a :class:`TetMeshSourceComplex` declares the
    authoritative rows, their deviation bounds and the initial witnesses. A
    face edge that is not a segment must be shared by exactly two faces of one
    source, coplanar unless a vertex carries a bounded witness. A ``"fixed"``
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

    def __init__(
        self,
        points: object,
        tetrahedra: object,
        tetrahedron_regions: object,
        faces: object,
        face_sources: object,
        segments: object,
        segment_sources: object,
        /,
        *,
        boundary_policy: TetMeshBoundaryPolicy,
        protection_radii: object | None = None,
        max_vertices: int | None = None,
        max_tetrahedra: int | None = None,
        max_scratch_bytes: int | None = None,
        source: TetMeshSourceComplex | None = None,
    ) -> None:
        if source is not None and not isinstance(source, TetMeshSourceComplex):
            raise TypeError("source must be TetMeshSourceComplex or None.")
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
                raise ValueError(f"protection_radii must have shape ({count},).")
        match boundary_policy:
            case "fixed":
                policy = 0
            case "conforming":
                policy = 1
            case _:
                raise ValueError(f"Unknown boundary policy {boundary_policy!r}.")
        vertices = _limit(max_vertices, _MAX_POINTS, "max_vertices")
        if vertices < count or vertices > _MAX_POINTS:
            raise ValueError("max_vertices must lie in [n, 2**31 - 2].")
        tetrahedra_limit = _limit(max_tetrahedra, 2**30 - 1, "max_tetrahedra")
        declared = None if source is None else _source_complex(source, count)
        face_ids = np.unique(sources) if source is None else source.face_ids
        segment_ids = np.unique(segment_labels) if source is None else source.segment_ids
        face_group_rows = _source_group_rows(sources, face_ids, "face_sources")
        segment_group_rows = _source_group_rows(
            segment_labels, segment_ids, "segment_sources"
        )
        handle = ctypes.c_void_p()
        status = library["phx_mc_tet_mesh_create"](
            count,
            point_array.ctypes.data,
            cells.shape[0],
            cells.ctypes.data,
            regions.ctypes.data,
            face_rows.shape[0],
            face_rows.ctypes.data,
            face_group_rows.ctypes.data,
            segment_rows.shape[0],
            segment_rows.ctypes.data,
            segment_group_rows.ctypes.data,
            _pointer(radii),
            0 if declared is None else declared[9].shape[0],
            None if declared is None else declared[9].ctypes.data,
            0 if declared is None else declared[0].shape[0],
            None if declared is None else declared[0].ctypes.data,
            None if declared is None else declared[1].ctypes.data,
            None if declared is None else declared[2].ctypes.data,
            0 if declared is None else declared[3].shape[0],
            None if declared is None else declared[3].ctypes.data,
            None if declared is None else declared[4].ctypes.data,
            None if declared is None else declared[5].ctypes.data,
            None if declared is None else declared[6].ctypes.data,
            None if declared is None else declared[7].ctypes.data,
            None if declared is None else declared[8].ctypes.data,
            policy,
            vertices,
            tetrahedra_limit,
            _budget(max_scratch_bytes, "max_scratch_bytes"),
            ctypes.addressof(handle),
        )
        _raise_for_call(status, "tet_mesh_create")
        if not handle.value:
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "tet_mesh_create: missing handle"
            )
        self._library = library
        self._handle = handle.value
        self._face_source_ids, self._segment_source_ids = face_ids, segment_ids
        face_ids.setflags(write=False)
        segment_ids.setflags(write=False)
        self._finalizer = weakref.finalize(self, _free_tet_mesh, library, handle.value)

    def _live(self) -> int:
        if not self._finalizer.alive:
            raise ValueError("the tetrahedral mesh is closed.")
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
        the original absolute ceiling. :meth:`proposal_shape` additionally caps
        its own query work without replacing or resetting that shared allowance.
        """
        if work_limit is None:
            raise TypeError("work_limit must be an integer.")
        _raise_for_call(
            self._library["phx_mc_tet_mesh_set_work_limit"](
                self._live(), _budget(work_limit, "work_limit")
            ),
            "tet_mesh_set_work_limit",
        )

    def work_units(self) -> int:
        """Return the actual cumulative native work consumed by this mesh."""
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
        record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
    ) -> TetMeshRun:
        counters = np.zeros((len(TET_MESH_REFINE_COUNTERS),), dtype=np.int64)
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
                        MeshcoreStatus.INTERNAL_ERROR, "unmeasured refinement invocation"
                    )
                recorded_work = (
                    int(counters[7])
                    if status in (MeshcoreStatus.OK, MeshcoreStatus.REFINEMENT_LIMIT)
                    else None
                )
                record_native_phase("refinement", float(seconds[0]), recorded_work, 1)
            return TetMeshRun(_run_status(status, "tet_mesh_refine"), counters)
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
        record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
    ) -> TetMeshRun:
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
            raise ValueError("radius_edge_bound must be positive.")
        insertions = _budget(max_insertions, "max_insertions")
        work = _budget(work_limit, "work_limit")
        if not callable(sizes):
            targets = None
            if sizes is not None:
                targets = _real_array(sizes, "sizes")
                if targets.shape != (int(self._counts()[0]),):
                    raise ValueError("sizes must have one value per vertex.")
            return self._refine_once(
                targets, radius_edge_bound, insertions, work, record_native_phase
            )
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive.")
        values = np.zeros((0,), dtype=np.float64)
        total = np.zeros((len(TET_MESH_REFINE_COUNTERS),), dtype=np.int64)
        run = TetMeshRun(MeshcoreStatus.OK, total)
        for _ in range(max_rounds):
            snapshot = self.arrays()
            fresh = snapshot.points[values.shape[0] :]
            sampled = _real_array(sizes(fresh), "sizes")
            if sampled.shape != (fresh.shape[0],):
                raise ValueError("the size callable must return one value per point.")
            values = np.concatenate((values, sampled))
            run = self._refine_once(
                values, radius_edge_bound, insertions, work, record_native_phase
            )
            # Work and insertion counters add up; the unmet count and vertex
            # count describe the last round, the largest cavity the maximum.
            total = total + run.counters
            total[6] = run.counters[6]
            total[8] = max(total[8] - run.counters[8], run.counters[8])
            total[9] = run.counters[9]
            added = int(run.counters[0] + run.counters[1] + run.counters[2])
            insertions -= added
            work -= int(run.counters[7])
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
    ) -> TetMeshRun:
        """Remove slivers by face/edge/multiface reconnection, source-preserving
        relocation and unprotected interior-vertex removal.

        Constrained faces, segments and regions never change; each operation
        is committed only when it raises the local smallest dihedral angle.
        ``minimum_relative_determinant`` is the owning validity policy's
        relative determinant floor: a cell whose exact determinant does not
        exceed it times the product of its canonical edge lengths is repaired
        under an exact construction objective, or reported with the
        ``validity`` criterion. ``0`` requires only exact positive orientation.
        """

        if not 0.0 <= min_dihedral_degrees < 70.5:
            raise ValueError("min_dihedral_degrees must lie in [0, 70.5).")
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError("minimum_relative_determinant must lie in [0, 1).")
        if isinstance(max_passes, bool) or max_passes < 0:
            raise ValueError("max_passes must be a nonnegative integer.")
        counters = np.zeros((len(TET_MESH_IMPROVE_COUNTERS),), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_improve"](
            self._live(),
            min_dihedral_degrees,
            minimum_relative_determinant,
            max_passes,
            _budget(work_limit, "work_limit"),
            counters.ctypes.data,
        )
        return TetMeshRun(_run_status(status, "tet_mesh_improve"), counters)

    def _applied(self, status: int, applied: np.ndarray, operation: str, /) -> bool:
        _raise_for_call(status, operation)
        return bool(applied[0])

    def flip_face(self, face: object, /, *, require_improvement: bool = False) -> bool:
        """2-3 flip of the unconstrained interior face ``(3,)``; whether applied."""

        vertices = _power_indices(face, "face")
        if vertices.shape != (3,):
            raise ValueError("face must have shape (3,).")
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
        """Move an unprotected interior, facet or exact segment-interior vertex.

        Conforming constrained motion preserves every incident source plane or
        segment interval and recertifies its witness before the atomic write.
        Automatic improvement does not generate segment-vertex moves.
        """

        target = _real_array(position, "position")
        if target.shape != (3,):
            raise ValueError("position must have shape (3,).")
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
        vertices: object,
        positions: object,
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
            raise ValueError("vertices and positions must have shapes (n,) and (n, 3).")
        if np.any(np.diff(rows) <= 0):
            raise ValueError("vertices must be sorted and unique.")
        if not np.isfinite(radius_edge_bound) or radius_edge_bound < 0.0:
            raise ValueError("radius_edge_bound must be finite and nonnegative.")
        if (
            not np.isfinite(minimum_dihedral_degrees)
            or not 0.0 <= minimum_dihedral_degrees < 180.0
        ):
            raise ValueError("minimum_dihedral_degrees must be finite and in [0, 180).")
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
        """Change the exact native allocation cap; refusal preserves the old cap."""
        _raise_for_call(
            self._library["phx_mc_tet_mesh_set_memory_limit"](
                self._live(), _budget(max_scratch_bytes, "max_scratch_bytes")
            ),
            "tet_mesh_set_memory_limit",
        )

    def memory_evidence(self) -> NDArray[np.uint64]:
        """Actual limit/live/peak/requested bytes and allocation/refusal counts."""
        values = np.empty((6,), dtype=np.uint64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_memory_evidence"](
                self._live(), values.ctypes.data
            ),
            "tet_mesh_memory_evidence",
        )
        return values

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

        ``source_fraction`` carries the actual parameter returned by
        :meth:`edge_split_point`. If an exact incident construction is absent,
        it permits only the identical recertified source-row RNE carrier under
        the original row's bound, never a snapped point or a changed stratum.
        """
        fraction = 0.0 if source_fraction is None else float(source_fraction)
        if source_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "source_fraction must be finite and strictly between zero and one."
            )
        point = _real_array(position, "position")
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position must be a finite three-coordinate point.")
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError("target_size must be finite and nonnegative.")
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

        The split witness must be exactly collinear and strictly inside the old
        edge. The final point preserves the actual scientific curve interval or
        exact oriented source-plane/material patch union, and the closed star
        shell. Only final positive cells are published; shape policy belongs to
        the caller. Local refusal returns None without an intermediate vertex.
        """
        witness = _real_array(split_position, "split_position")
        if witness.shape != (3,) or not np.all(np.isfinite(witness)):
            raise ValueError("split_position must be a finite three-coordinate point.")
        final = _real_array(final_position, "final_position")
        if final.shape != (3,) or not np.all(np.isfinite(final)):
            raise ValueError("final_position must be a finite three-coordinate point.")
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError("target_size must be finite and nonnegative.")
        inserted = np.full((1,), -1, dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_split_edge_relocate"](
            self._live(),
            _node_id(first, "first"),
            _node_id(second, "second"),
            witness.ctypes.data,
            final.ctypes.data,
            target_size,
            _budget(work_limit, "work_limit"),
            inserted.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_split_edge_relocate")
        return None if inserted[0] < 0 else int(inserted[0])

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
    ) -> TetMeshInsertionPlan | None:
        """Inspect the exact expanded final insertion cavity without publishing.

        Source/material/ghost reconstruction belongs to the existing native
        constrained preparation. Finite removed+proposed rows share the given
        original cavity allowance. Refusals leave accepted geometry unchanged.
        ``source_fraction`` authenticates the original bounded source witness
        exactly as in :meth:`split_edge`; inspection and commit retain the same
        fraction, RNE carrier and complete native conflict cavity.
        """
        witness = _real_array(split_position, "split_position")
        position = _real_array(final_position, "final_position")
        if witness.shape != (3,) or not np.all(np.isfinite(witness)):
            raise ValueError("split_position must be a finite three-coordinate point.")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("final_position must be a finite three-coordinate point.")
        fraction = 0.0 if source_fraction is None else float(source_fraction)
        if source_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "source_fraction must be finite and strictly between zero and one."
            )
        maximum = _budget(maximum_cavity_cells, "maximum_cavity_cells")
        if maximum > np.iinfo(np.intp).max // (8 * np.dtype(np.int32).itemsize):
            raise ValueError("The insertion cavity buffers must be addressable.")
        budget = current_native_execution_budget()
        if budget is None:
            removed = np.empty((maximum, 4), dtype=np.int32)
            proposed = np.empty((maximum, 4), dtype=np.int32)
        else:
            removed = budget.allocate_host_array((maximum, 4), np.int32)
            proposed = budget.allocate_host_array((maximum, 4), np.int32)
        counts = np.zeros((4,), dtype=np.int64)
        status = self._library["phx_mc_tet_mesh_inspect_edge_insertion"](
            self._live(),
            _node_id(first, "first"),
            _node_id(second, "second"),
            witness.ctypes.data,
            position.ctypes.data,
            fraction,
            maximum,
            _budget(work_limit, "work_limit"),
            removed.ctypes.data,
            proposed.ctypes.data,
            counts.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_inspect_edge_insertion")
        old_count, new_count, kind, generation = (int(value) for value in counts)
        if old_count == 0 or new_count == 0:
            return None
        if old_count < 0 or new_count < 0 or old_count + new_count > maximum:
            raise RuntimeError(
                "Native insertion inspection exceeded its admitted cavity extent."
            )
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
            ),
        )

    def commit_edge_insertion(
        self,
        plan: TetMeshInsertionPlan,
        /,
        *,
        target_size: float = 0.0,
        work_limit: int | None = None,
    ) -> int | None:
        """Reprepare this exact final source witness/cavity and commit once.

        A foreign owner, stale accepted-state generation or changed inspected
        tuples never becomes a native reconstruction or a partial publication.
        """
        if not isinstance(plan, TetMeshInsertionPlan):
            raise TypeError("plan must be TetMeshInsertionPlan.")
        if plan._owner is not self:
            raise ValueError("An insertion plan must belong to this native owner.")
        if not np.isfinite(target_size) or target_size < 0.0:
            raise ValueError("target_size must be finite and nonnegative.")
        inserted = np.full((1,), -1, dtype=np.int32)
        status = self._library["phx_mc_tet_mesh_commit_edge_insertion"](
            self._live(),
            plan.first,
            plan.second,
            plan.split_position.ctypes.data,
            plan.position.ctypes.data,
            plan.source_fraction,
            plan.maximum_cavity_cells,
            plan.insertion_kind,
            plan.accepted_generation,
            plan.removed.shape[0],
            plan.removed.ctypes.data,
            plan.proposed.shape[0],
            plan.proposed.ctypes.data,
            target_size,
            _budget(work_limit, "work_limit"),
            inserted.ctypes.data,
        )
        _raise_for_call(status, "tet_mesh_commit_edge_insertion")
        return None if inserted[0] < 0 else int(inserted[0])

    def protect_vertices(self, vertices: object, /) -> None:
        """Lock explicitly protected live vertices atomically by their exact IDs."""
        ids = _power_indices(vertices, "vertices")
        if ids.ndim != 1 or np.any(ids < 0):
            raise ValueError("vertices must be a nonnegative integer vector.")
        _raise_for_call(
            self._library["phx_mc_tet_mesh_protect_vertices"](
                self._live(), ids.shape[0], ids.ctypes.data
            ),
            "tet_mesh_protect_vertices",
        )

    def construct_curve_points(
        self,
        vertices: object,
        neighbors: object,
        desired_positions: object,
        /,
        *,
        work_limit: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Construct a batch on actual source-curve intervals without mutation.

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
                "Curve vertices, neighbors and desired positions must align as (n,), (n, 2), (n, 3)."
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
        """Construct an incident exact point or a bounded source-row RNE carrier.

        None retains the original dyadic construction order. An explicit finite
        fraction strictly between zero and one is attempted exactly, then exact
        dyadic candidates are ranked by distance if that attempt is refused.
        When no exact candidate exists, the owning source construction tries the
        preferred fraction (or one half) under its declared row bound. Pass the
        returned actual fraction to :meth:`split_edge` as ``source_fraction``;
        no source witness is inferred from rounded coordinates. None means no
        candidate was admitted, never an approximate source snap.
        """
        fraction = 0.0 if preferred_fraction is None else float(preferred_fraction)
        if preferred_fraction is not None and (
            not np.isfinite(fraction) or not 0.0 < fraction < 1.0
        ):
            raise ValueError(
                "preferred_fraction must be finite and strictly between zero and one."
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
        faces = np.empty((int(counts[2]),), dtype=np.int32)
        segments = np.empty((int(counts[3]),), dtype=np.int32)
        strata = np.empty((vertices,), dtype=np.int8)
        entities = np.empty((vertices,), dtype=np.int32)
        parameters = np.empty((vertices, 2), dtype=np.float64)
        deviations = np.empty((vertices,), dtype=np.float64)
        _raise_for_call(
            self._library["phx_mc_tet_mesh_source_evidence"](
                self._live(),
                values.ctypes.data,
                faces.ctypes.data,
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
            faces,
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
            faces,
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
            raise TypeError("enabled must be boolean.")
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
                "removed and proposed must be nonempty tetrahedral vertex tuples."
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
            raise ValueError("cells must be a nonempty array of four-vertex tuples.")
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
    ) -> tuple[TetMeshRun, NDArray[np.float64]]:
        """Explicit bounded local weighted sliver stage, not global regularity.

        Only unprotected interior vertices receive power weights below
        ``max_weight_fraction`` times their shortest incident edge squared;
        weighted 2-3 and ring reconnections must raise the local smallest
        dihedral angle and keep size targets. No reconnection creates a cell at
        or below ``minimum_relative_determinant`` on its canonical chart (the
        publishing validity floor); remaining cells are :meth:`unmet` records.
        Counters follow :data:`TET_MESH_EXUDE_COUNTERS`; the weights are the
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
                "weighted improvement requires finite admissible quality bounds."
            )
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError("minimum_relative_determinant must lie in [0, 1).")
        passes = _budget(max_passes, "max_passes")
        if passes > np.iinfo(np.int32).max:
            raise ValueError("max_passes must fit signed int32.")
        weights = np.empty((int(self._counts()[0]),), dtype=np.float64)
        counters = np.zeros((len(TET_MESH_EXUDE_COUNTERS),), dtype=np.int64)
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
        return TetMeshRun(_run_status(status, "tet_mesh_exude"), counters), weights

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
        points: object,
        tetrahedra: object,
        /,
        *,
        vertex_ids: object,
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
        ``below_floor`` counts cells whose exact determinant does not exceed
        ``minimum_relative_determinant`` times the product of the edge lengths
        from the cell's canonical (smallest-identity) vertex: the publishing
        :class:`~phydrax.discretization.CellValidityPolicy` floor semantics.

        Coordinate-validation visits, index visits, cell shape and floor
        evaluations consume the owner's actual cumulative :meth:`work_units`.
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
            raise ValueError("vertex_ids must have one identity per point.")
        if np.any(raw_identities < 0) or (
            raw_identities.size and np.max(raw_identities) > np.iinfo(np.int64).max
        ):
            raise ValueError("vertex_ids must be nonnegative int64 identities.")
        identities = np.ascontiguousarray(raw_identities, dtype=np.int64)
        if not 0.0 <= minimum_relative_determinant < 1.0:
            raise ValueError("minimum_relative_determinant must lie in [0, 1).")
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
        """Canonical host snapshot; the mesh stays usable."""

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
@dataclass(frozen=True, slots=True)
class RestrictedPowerCells:
    """Native restricted power cells and decomposition lineage on the host.

    Face loops point outward from the first cell; a negative second cell encodes
    ``-(1 + boundary_facet)``. Second moments are scalar integrals of squared
    distance to the owning site, not full covariance tensors.
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
    counters: NDArray[np.int64]
    vertex_carriers: NDArray[np.int32]
    vertex_site_offsets: NDArray[np.int64]
    vertex_sites: NDArray[np.int32]
    source_work_units: int


@final
class RestrictedPowerFailure(MeshcoreError):
    """Native refusal retaining the failure location and consumed work."""

    def __init__(
        self,
        status: MeshcoreStatus,
        failure: NDArray[np.int64],
        counters: NDArray[np.int64],
        /,
    ) -> None:
        super().__init__(status, f"restricted_power_cells: {tuple(failure)}")
        self.failure = failure
        self.counters = counters


def _power_indices(value: object, name: str, /) -> NDArray[np.int32]:
    values = _index_array(value, name)
    if np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max):
        raise ValueError(f"{name} must fit signed int32.")
    return np.ascontiguousarray(values, dtype=np.int32)


def _copy_power_cells(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    counters: NDArray[np.int64],
    /,
) -> RestrictedPowerCells:
    sizes = np.zeros((5,), dtype=np.int64)
    library["phx_mc_power_cells_sizes"](handle, sizes.ctypes.data)
    source_work_units = library["phx_mc_power_cells_source_work_units"](handle)
    if source_work_units < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "negative power-cell source work"
        )
    if np.any(sizes < 0):
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "negative power-cell sizes")
    vertices, faces, entries, cells, pieces = (int(size) for size in sizes)
    witness_count = int(library["phx_mc_power_cells_site_witness_size"](handle))
    if witness_count < 0:
        raise MeshcoreError(
            MeshcoreStatus.INTERNAL_ERROR, "negative power-cell site witness size"
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
        counters,
        np.empty((vertices, 4), dtype=np.int32),
        np.empty((vertices + 1,), dtype=np.int64),
        np.empty((witness_count,), dtype=np.int32),
        source_work_units,
    )
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
    )
    library["phx_mc_power_cells_export_vertex_carriers"](
        handle, result.vertex_carriers.ctypes.data
    )
    result.vertex_carriers.flags.writeable = False
    library["phx_mc_power_cells_export_vertex_sites"](
        handle,
        result.vertex_site_offsets.ctypes.data,
        result.vertex_sites.ctypes.data,
    )
    result.vertex_site_offsets.flags.writeable = False
    result.vertex_sites.flags.writeable = False
    return result


def restricted_power_cells(
    sites: object,
    weights: object,
    neighbor_offsets: object,
    neighbors: object,
    split_sites: object,
    points: object,
    tets: object,
    tet_regions: object,
    tet_face_facets: object,
    /,
    *,
    max_pieces: int,
    max_vertices: int,
    work_limit: int,
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
) -> RestrictedPowerCells:
    """Clip weighted cells to an authoritative tetrahedral domain.

    Neighbor CSR comes from the native regular triangulation. Domain cells
    must be positively oriented with every exterior face explicitly tagged.
    A native failure never returns partial cells as an accepted decomposition.
    """

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
        raise ValueError("weights must be a finite vector with one value per site.")
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or adjacent.ndim != 1
        or offsets[-1] != adjacent.size
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(adjacent < 0)
        or np.any(adjacent >= count)
    ):
        raise ValueError("neighbor_offsets and neighbors must define site-indexed CSR.")
    if split.shape != (count,) or np.any((split != 0) & (split != 1)):
        raise ValueError("split_sites must contain one zero-or-one flag per site.")
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or regions.shape != (tetrahedra.shape[0],)
        or facets.shape != tetrahedra.shape
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= point_array.shape[0])
    ):
        raise ValueError(
            "tets, tet_regions and tet_face_facets must align with domain points."
        )
    flags = np.ascontiguousarray(split, dtype=np.int8)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_restricted_power_cells"](
            count,
            site_array.ctypes.data,
            weight_array.ctypes.data,
            offsets.ctypes.data,
            adjacent.ctypes.data,
            flags.ctypes.data,
            point_array.shape[0],
            point_array.ctypes.data,
            tetrahedra.shape[0],
            tetrahedra.ctypes.data,
            regions.ctypes.data,
            facets.ctypes.data,
            _limit(max_pieces, 1, "max_pieces"),
            _limit(max_vertices, 1, "max_vertices"),
            _limit(work_limit, 1, "work_limit"),
            int(record_native_phase is not None),
            ctypes.addressof(handle),
        )
    )
    return _finish_power_cells(library, handle, status, record_native_phase)


def _finish_power_cells(
    library: MeshcoreLibrary,
    handle: ctypes.c_void_p,
    status: MeshcoreStatus,
    record_native_phase: Callable[[str, float, int | None, int], None] | None,
    /,
) -> RestrictedPowerCells:
    """Own every native export and failure record until the result is freed."""
    if not handle.value:
        raise MeshcoreError(
            status if status != MeshcoreStatus.OK else MeshcoreStatus.INTERNAL_ERROR,
            "missing power-cell result",
        )
    # Own the native result until all exports finish, including failure evidence.
    try:
        counters = np.zeros((10,), dtype=np.int64)
        library["phx_mc_power_cells_counters"](handle, counters.ctypes.data)
        if record_native_phase is not None:
            seconds = np.empty((4,), dtype=np.float64)
            library["phx_mc_power_cells_phase_seconds"](handle, seconds.ctypes.data)
            if not np.all(np.isfinite(seconds)):
                raise MeshcoreError(
                    MeshcoreStatus.INTERNAL_ERROR, "invalid power-cell timings"
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
            library["phx_mc_power_cells_failure"](handle, failure.ctypes.data)
            raise RestrictedPowerFailure(status, failure, counters)
        return _copy_power_cells(library, handle, counters)
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
    record_native_phase: Callable[[str, float, int | None, int], None] | None = None,
) -> RestrictedPowerCells:
    """Clip cells from exact image coordinates and their original site owners."""
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
            "weights must be a finite vector with one value per original site."
        )
    if (
        owners.ndim != 1
        or owners.size == 0
        or owners.size > _MAX_POINTS
        or np.any(owners < 0)
        or np.any(owners >= original.shape[0])
    ):
        raise ValueError("image_site_owners must identify every image's original site.")
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
            "Exact image coordinate offsets must bind complete finite expansion components."
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
        raise ValueError("neighbor_offsets and neighbors must define image-indexed CSR.")
    if split.shape != (image_count,) or np.any((split != 0) & (split != 1)):
        raise ValueError("split_sites must contain one zero-or-one flag per image.")
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or regions.shape != (tetrahedra.shape[0],)
        or facets.shape != tetrahedra.shape
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= domain.shape[0])
    ):
        raise ValueError(
            "tets, tet_regions and tet_face_facets must align with domain points."
        )
    flags = np.ascontiguousarray(split, dtype=np.int8)
    library = load_meshcore()
    handle = ctypes.c_void_p()
    status = MeshcoreStatus(
        library["phx_mc_restricted_power_cells_exact"](
            original.shape[0],
            original.ctypes.data,
            weight_array.ctypes.data,
            image_count,
            owners.ctypes.data,
            coordinate_offsets.ctypes.data,
            components.size,
            components.ctypes.data,
            offsets.ctypes.data,
            adjacent.ctypes.data,
            flags.ctypes.data,
            domain.shape[0],
            domain.ctypes.data,
            tetrahedra.shape[0],
            tetrahedra.ctypes.data,
            regions.ctypes.data,
            facets.ctypes.data,
            _limit(max_pieces, 1, "max_pieces"),
            _limit(max_vertices, 1, "max_vertices"),
            _limit(work_limit, 1, "work_limit"),
            int(record_native_phase is not None),
            ctypes.addressof(handle),
        )
    )
    return _finish_power_cells(library, handle, status, record_native_phase)


ARRANGEMENT_UNCLASSIFIED = int(np.iinfo(np.int32).min)


@final
@dataclass(frozen=True, slots=True)
class ArrangementClassification:
    """Exact operand membership of arrangement fragments.

    Fragments of one operand are joined across shared edges that are not
    contact edges.  ``component_windings[c, j]`` is the exact winding number of
    operand ``j`` at component ``c`` (``fragment_components`` maps fragments to
    components), or ``ARRANGEMENT_UNCLASSIFIED`` for the component's own
    operand and operands coplanar with it.  ``status`` is ``OK`` or
    ``CAPACITY_EXCEEDED``; a refused classification has empty arrays.
    ``pair_scans`` (representative/triangle pairs) and ``matrix_entries`` are
    the admitted work, or on refusal the requirement evaluated before it was
    refused; ``components`` is the component count.
    """

    status: MeshcoreStatus
    fragment_components: NDArray[np.int64]
    component_windings: NDArray[np.int32]
    components: int
    pair_scans: int
    matrix_entries: int


@final
@dataclass(frozen=True, slots=True)
class NativeArrangement:
    """Exact construction identities with bounded host-coordinate publication."""

    vertices: NDArray[np.float64]
    bounds: NDArray[np.float64]
    constructions: NDArray[np.int8]
    origins: NDArray[np.int64]
    fragments: NDArray[np.int64]
    fragment_faces: NDArray[np.int64]
    contact_edges: NDArray[np.int64]
    coplanar: NDArray[np.int64]
    locations: NDArray[np.int64]
    counters: NDArray[np.int64]
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
    library["phx_mc_arrangement_sizes"](handle, sizes.ctypes.data)
    if np.any(sizes < 0):
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "negative arrangement sizes")
    vertices, fragments, edges, coplanar, locations = (int(size) for size in sizes)
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
        np.empty((11,), dtype=np.int64),
        None,
    )
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
    )
    library["phx_mc_arrangement_counters"](handle, result.counters.ctypes.data)
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
            library["phx_mc_arrangement_export_classification"](
                handle, fragment_components.ctypes.data, component_windings.ctypes.data
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
    vertices: object,
    triangles: object,
    surfaces: object,
    pairs: object,
    /,
    *,
    maximum_vertices: int,
    maximum_fragments: int,
    classification_operands: int | None = None,
    classification_work_limit: int | None = None,
) -> NativeArrangement:
    """Arrange all declared cross-operand triangle contacts simultaneously.

    Candidate pairs are explicit bounded input from the owning spatial route.
    The native indirect predicates retain original construction identity.
    With ``classification_operands`` (the declared dense operand count of
    ``surfaces``) and ``classification_work_limit`` the exact operand
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
        raise ValueError("triangles, surfaces and pairs must align with input vertices.")
    vertex_limit = _limit(maximum_vertices, points.shape[0], "maximum_vertices")
    fragment_limit = _limit(maximum_fragments, faces.shape[0], "maximum_fragments")
    if vertex_limit < points.shape[0] or fragment_limit < faces.shape[0]:
        raise ValueError("arrangement capacities must include the input geometry.")
    if (classification_operands is None) != (classification_work_limit is None):
        raise ValueError(
            "classification_operands and classification_work_limit are given together."
        )
    classification_request = None
    if classification_operands is not None and classification_work_limit is not None:
        work_limit = _unsigned_budget(
            classification_work_limit, "classification_work_limit"
        )
        if work_limit > np.iinfo(np.int64).max:
            raise ValueError("classification_work_limit must fit a signed int64.")
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
                MeshcoreStatus.INTERNAL_ERROR, "missing arrangement result"
            )
        result = _copy_arrangement(library, handle)
        if classification_request is None:
            return result
        return replace(
            result,
            classification=_classify_arrangement(
                library, handle, result.fragments.shape[0], *classification_request
            ),
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
        raise ValueError(f"nodes must have shape (cells, {width}).")
    if np.any(data < 0) or np.any(data >= count):
        raise ValueError("nodes must index the declared vertex table.")
    capacity = _limit(maximum_cells, 1, "maximum_cells")
    needed = multiplier * data.shape[0]
    if needed > capacity:
        raise MeshcoreError(MeshcoreStatus.CAPACITY_EXCEEDED, f"{name}: cell budget")
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
    """Four genuine dual hexes per tetrahedron from shared fifteen-node tuples."""
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
    vertices: object,
    faces: object,
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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """Explicit bounded wrapping with the same authoritative root skin and cut cells."""
    points = _points(vertices, "vertices", 3)
    triangles = np.ascontiguousarray(_index_array(faces, "faces"), dtype=np.int64)
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or np.any(triangles < 0)
        or np.any(triangles >= points.shape[0])
    ):
        raise ValueError("faces must contain three valid vertex indices per triangle.")
    parameters = np.asarray((offset, spacing, certificate_spacing), dtype=np.float64)
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError(
            "offset, spacing and certificate_spacing must be finite positive."
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
        points.ctypes.data,
        points.shape[0],
        triangles.ctypes.data,
        triangles.shape[0],
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        _limit(max_samples, 1, "max_samples"),
        _limit(max_work_units, 1, "max_work_units"),
        *capacities,
        ctypes.addressof(handle),
        counts.ctypes.data,
        bounds.ctypes.data,
    )
    try:
        if status == MeshcoreStatus.RANGE_ERROR:
            raise ValueError(
                "surface_envelope: a finite source-inclusion certificate is unavailable "
                "at the declared coordinate and sampling scales."
            )
        _raise_for_call(status, "surface_envelope")
        if not handle.value:
            raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "missing envelope result")
        sizes = (int(counts[0]), int(counts[1]), int(counts[5]), int(counts[6]))
        if any(
            size < 0 or size > limit
            for size, limit in zip(sizes, capacities, strict=True)
        ):
            raise MeshcoreError(
                MeshcoreStatus.INTERNAL_ERROR, "invalid envelope output counts"
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
        return surface_points, surface_faces, counts, bounds, volume_points, cells
    finally:
        if handle.value:
            library["phx_mc_surface_envelope_free"](handle)


def restricted_dual_3d(
    points: object,
    tets: object,
    domain: object,
    max_facets: int,
    work_limit: int,
    /,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int64],
]:
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
        raise ValueError("tets and domain must describe a three-dimensional point set.")
    bounds = np.ascontiguousarray(bounds.reshape((6,)), dtype=np.float64)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("domain bounds must be finite and ordered.")
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
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "invalid restricted dual size")
    n = count[0]
    return facets[:n], incident[:n], endpoints[:n], kinds[:n], items[:n], counters


def restricted_centers_3d(
    points: object,
    tetrahedra: object,
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
        raise ValueError("tetrahedra must contain valid four-vertex tuples.")
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
                "Mixed templates require a quadrilateral, tetrahedron, hexahedron, prism or pyramid."
            )
    if not isinstance(axial, (bool, np.bool_)):
        raise TypeError("axial must be boolean.")
    library = load_meshcore()
    counts = np.zeros((3,), dtype=np.int32)
    _raise_for_call(
        library["phx_mc_mixed_template_counts"](kind, int(axial), 0, counts.ctypes.data),
        "mixed_template_counts",
    )
    if counts[0] < 1:
        raise MeshcoreError(MeshcoreStatus.INTERNAL_ERROR, "missing mixed templates")
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
                MeshcoreStatus.INTERNAL_ERROR, "invalid mixed-template sizes"
            )
        shape = (int(counts[1]), int(counts[2]), 2 if kind == 4 else 3)
        budget = current_native_execution_budget()
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
    points: object,
    tetrahedra: object,
    center_bounds: object,
    facets: object,
    incident_cells: object,
    domain: object,
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
            "ray facets, cells and center bounds must align with tetrahedra."
        )
    box = np.ascontiguousarray(box.reshape((6,)), dtype=np.float64)
    if not np.all(np.isfinite(box)) or np.any(box[3:] <= box[:3]):
        raise ValueError("domain bounds must be finite and ordered.")
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
    "current_native_execution_budget",
    "charge_native_geometry_queries",
    "NativeExecutionBudget",
    "NativeExecutionAllowance",
    "NativeExecutionEvidence",
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
    "PLC_3D_COUNTERS",
    "PLC_3D_ENTITY_KINDS",
    "PLC_3D_FAILURE_REASONS",
    "PlcRecovery3D",
    "PlcRecoveryFailure",
    "PlcSourceConstraints",
    "TetMeshBoundaryPolicy",
    "TET_MESH_EXUDE_COUNTERS",
    "TET_MESH_IMPROVE_COUNTERS",
    "TET_MESH_REFINE_COUNTERS",
    "TET_MESH_UNMET_CRITERIA",
    "TET_MESH_UNMET_REASONS",
    "TetMesh3D",
    "TetMeshArrays",
    "TetMeshInsertionPlan",
    "TetMeshEdgeConstruction",
    "TetMeshQuality",
    "TetMeshRun",
    "TetMeshSourceComplex",
    "TetMeshSourceEvidence",
    "TetMeshUnmet",
    "RestrictedPowerCells",
    "RestrictedPowerFailure",
    "GRAPH_PARTITION_COUNTERS",
    "PERIODIC_DELAUNAY_EVIDENCE",
    "SURFACE_RECONNECTION_COUNTERS",
    "TRIANGULATION_3D_STATISTICS",
    "mixed_refinement_templates",
    "restricted_centers_3d",
    "restricted_rays_3d",
    "clip_box_halfplanes",
    "clip_reference_triangle_exact",
    "ExactReferenceTetrahedronClip",
    "clip_reference_tetrahedron_exact",
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
    "symbolic_orient3d",
    "surface_reconnect",
    "tetrahedron_intersection_moments",
    "tetrahedron_intersection_simplices",
    "triangle_intersection_classes",
    "triangle_intersections",
]
