#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Site-MPI ownership and host admission for ordered numerical collectives.

The initialized execution group is a declared reservation, not evidence of an
OS-exclusive allocation. MPI initialization and the bounded process supervisor
belong to bootstrap. This module never imports mpi4py or initializes MPI.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final, Protocol

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding
from jax.stages import Compiled, Traced

from . import _jax_config
from ._cuda_native import cuda_component_function
from ._execution_device import device_platform
from ._execution_resources import ExecutionGroupSpec
from ._execution_runtime import ExecutionGroup
from ._fingerprint import canonical_fingerprint
from ._jax_context import inside_jax_transformation
from ._strict import StrictModule


def _require_host_phase(operation: str) -> None:
    if inside_jax_transformation():
        raise ValueError(
            f"collective {operation} is a host phase and cannot run inside a JAX transformation"
        )


class CollectiveLeaseError(RuntimeError):
    """A collective owner refused admission or its coupled execution failed."""


_QUALIFIED_COLLECTIVE_JAX_RELEASE = "0.12.0"
_COLLECTIVE_PRIVATE_DISPATCH = (
    "jax.stages.Compiled.call/Compiled._params/pxla.ExecuteReplicated.runtime_tokens"
)


def _require_collective_jax_abi() -> None:
    if jax.__version__ != _QUALIFIED_COLLECTIVE_JAX_RELEASE:
        raise CollectiveLeaseError(
            "ordered collective execution requires the qualified JAX 0.12.0 "
            f"private dispatch ABI; found unqualified JAX {jax.__version__}"
        )


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty NUL-free identity")


@dataclass(frozen=True, slots=True)
class CollectiveLeaseSpec:
    """Stable group/provider/preparation identity, without a live native handle."""

    group_id: str
    provider_id: str
    configuration_id: str
    preparation_id: str
    process_indices: tuple[int, ...]
    device_keys: tuple[tuple[int, int], ...]
    mpi_build_identity: str

    def __post_init__(self) -> None:
        for name, value in (
            ("group_id", self.group_id),
            ("provider_id", self.provider_id),
            ("configuration_id", self.configuration_id),
            ("preparation_id", self.preparation_id),
            ("mpi_build_identity", self.mpi_build_identity),
        ):
            _identity(value, name)
        if not isinstance(self.process_indices, tuple) or not self.process_indices:
            raise ValueError(
                "collective process membership must be an immutable nonempty tuple"
            )
        if any(
            type(process) is not int or not 0 <= process <= np.iinfo(np.int32).max
            for process in self.process_indices
        ):
            raise ValueError(
                "collective process indices must be nonnegative int32 integers"
            )
        if len(set(self.process_indices)) != len(self.process_indices):
            raise ValueError("collective process membership must be unique")
        if not isinstance(self.device_keys, tuple) or len(self.device_keys) != len(
            self.process_indices
        ):
            raise ValueError(
                "MPI consumers require exactly one GPU per participating process"
            )
        if any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(type(value) is not int or value < 0 for value in key)
            for key in self.device_keys
        ):
            raise ValueError(
                "collective device keys must be immutable nonnegative integer pairs"
            )
        if tuple(key[0] for key in self.device_keys) != self.process_indices:
            raise ValueError(
                "collective device order must match one-GPU-per-rank process order"
            )

    @property
    def identity(self) -> str:
        return canonical_fingerprint(
            {
                "group_id": self.group_id,
                "provider_id": self.provider_id,
                "configuration_id": self.configuration_id,
                "preparation_id": self.preparation_id,
                "process_indices": self.process_indices,
                "device_keys": self.device_keys,
                "mpi_build_identity": self.mpi_build_identity,
            }
        )

    def require_group(self, group: ExecutionGroupSpec) -> None:
        if (
            group.group_id != self.group_id
            or group.process_indices != self.process_indices
            or group.device_keys != self.device_keys
        ):
            raise ValueError(
                "collective lease membership does not match its execution group"
            )


@final
class CollectiveBinding(StrictModule):
    """Dynamic lease generation/ticket leaves and stable preparation metadata."""

    lease_id: Array
    invocation_ticket: Array
    spec: CollectiveLeaseSpec = eqx.field(static=True)

    def __init__(
        self, lease_id: Array, invocation_ticket: Array, spec: CollectiveLeaseSpec
    ) -> None:
        for name, value in (
            ("lease_id", lease_id),
            ("invocation_ticket", invocation_ticket),
        ):
            if value.shape != () or value.dtype != np.dtype(np.int64):
                raise ValueError(f"collective {name} must be a scalar int64 array")
        self.lease_id = lease_id
        self.invocation_ticket = invocation_ticket
        self.spec = spec

    @property
    def group_id(self) -> str:
        return self.spec.group_id

    @property
    def provider_id(self) -> str:
        return self.spec.provider_id

    @property
    def configuration_id(self) -> str:
        return self.spec.configuration_id

    @property
    def preparation_id(self) -> str:
        return self.spec.preparation_id


@dataclass(frozen=True, slots=True)
class CollectiveMPIProvenance:
    """Exact bridge, site MPI library, header, and initialized runtime evidence."""

    build_identity: str
    library_path: str
    library_sha256: str
    header_sha256: str
    library_version: str


class CollectiveSupervisor(Protocol):
    """Bootstrap-owned bounded process-group failure coupling, not MPI consensus."""

    @property
    def supervisor_id(self) -> str: ...

    def require_active(
        self, process_indices: tuple[int, ...], timeout_seconds: float
    ) -> None: ...

    def fail(self, reason: str) -> None: ...


class _NativeBridge:
    """Only the canonical admitted site companion supplies these host symbols."""

    def function(self, symbol: str) -> Any:
        return cuda_component_function("collective", symbol)

    def check(self, status: int) -> None:
        if status:
            message = self.function("phx_collective_last_error")()
            if not isinstance(message, bytes):
                raise CollectiveLeaseError(
                    "site collective bridge returned no failure evidence"
                )
            raise CollectiveLeaseError(message.decode("utf-8"))

    def text(self, symbol: str) -> str:
        value = self.function(symbol)()
        if not isinstance(value, bytes) or not value:
            message = self.function("phx_collective_last_error")()
            detail = message.decode("utf-8") if isinstance(message, bytes) else symbol
            raise CollectiveLeaseError(f"site MPI provenance is unavailable: {detail}")
        return value.decode("utf-8")

    def provenance(self) -> CollectiveMPIProvenance:
        result = CollectiveMPIProvenance(
            self.text("phx_collective_build_identity"),
            self.text("phx_collective_mpi_library"),
            self.text("phx_collective_mpi_sha256"),
            self.text("phx_collective_mpi_header_sha256"),
            self.text("phx_collective_mpi_library_version"),
        )
        library = Path(result.library_path)
        if not library.is_absolute() or not library.is_file():
            raise CollectiveLeaseError(
                "site MPI provenance must identify an existing absolute library"
            )
        with library.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != result.library_sha256:
            raise CollectiveLeaseError(
                "site MPI library differs from its collective bridge build identity"
            )
        mpi: Any = sys.modules["mpi4py.MPI"]
        self._mpi_extension = ctypes.CDLL(mpi.__file__)
        symbol = ctypes.cast(self._mpi_extension.MPI_Get_library_version, ctypes.c_void_p)
        self.check(self.function("phx_collective_validate_mpi_symbol")(symbol))
        return result

    def create(
        self,
        spec: CollectiveLeaseSpec,
        source_fortran: int,
        process_index: int,
        source_ranks: tuple[int, ...],
        agreement_id: str,
        timeout_seconds: float,
    ) -> int:
        processes = (ctypes.c_int32 * len(spec.process_indices))(*spec.process_indices)
        ranks = (ctypes.c_int32 * len(source_ranks))(*source_ranks)
        result = ctypes.c_int64()
        group_identity = canonical_fingerprint(
            {
                "group_id": spec.group_id,
                "process_indices": spec.process_indices,
                "device_keys": spec.device_keys,
            }
        )
        self.check(
            self.function("phx_collective_create")(
                source_fortran,
                process_index,
                processes,
                ranks,
                len(source_ranks),
                int(group_identity[:8], 16) % 32768,
                agreement_id.encode(),
                spec.identity.encode(),
                timeout_seconds,
                ctypes.byref(result),
            )
        )
        if result.value <= 0:
            raise CollectiveLeaseError(
                "site MPI bridge returned an invalid native lease identity"
            )
        return result.value

    def validate(self, lease_id: int) -> None:
        self.check(self.function("phx_collective_validate")(lease_id))

    def agree_program(self, lease_id: int, schedule: tuple[str, ...]) -> None:
        self.check(
            self.function("phx_collective_agree_program")(
                lease_id,
                canonical_fingerprint(schedule).encode(),
            )
        )

    def begin(self, lease_id: int, schedule: tuple[str, ...]) -> int:
        ticket = ctypes.c_int64()
        encoded = "".join(schedule).encode()
        schedule_identity = canonical_fingerprint(schedule).encode()
        self.check(
            self.function("phx_collective_begin")(
                lease_id,
                encoded,
                len(schedule),
                schedule_identity,
                ctypes.byref(ticket),
            )
        )
        return ticket.value

    def end(self, lease_id: int, ticket: int) -> None:
        self.check(self.function("phx_collective_end")(lease_id, ticket))

    def fail(self, lease_id: int, reason: str) -> None:
        self.check(self.function("phx_collective_fail")(lease_id, reason.encode()))

    def retire(self, lease_id: int) -> None:
        self.check(self.function("phx_collective_retire")(lease_id))


_LIVE: dict[CollectiveLeaseSpec, CollectiveLease] = {}
_LIVE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class CollectiveProgramProvenance:
    """Qualified upstream release and the exact owned private dispatch boundary."""

    jax_release: str
    private_dispatch_identity: str

    @property
    def identity(self) -> str:
        return canonical_fingerprint(
            {
                "jax_release": self.jax_release,
                "private_dispatch_identity": self.private_dispatch_identity,
            }
        )


@final
@dataclass(frozen=True, slots=True, init=False)
class CollectiveProgram[Result]:
    """Host-owned executable and schedule derived from the exact same graph.

    This is deliberately not a PyTree or serializable numerical-plan field.
    Only explicit host staging may execute Python; invocation never retraces.
    """

    spec: CollectiveLeaseSpec
    schedule: tuple[str, ...]
    executable: Compiled
    provenance: CollectiveProgramProvenance

    def __init__(self, traced: Traced, *, lease_key: CollectiveLeaseSpec) -> None:
        _require_host_phase("program compilation")
        _require_collective_jax_abi()
        if type(traced) is not Traced:
            raise TypeError("CollectiveProgram requires an actual JAX Traced graph")
        lease = require_collective_lease(lease_key)
        constants, _ = traced.closure_convert()
        if jax.tree.leaves(constants):
            raise ValueError(
                "captured collective program numerical constants must be explicit dynamic arguments"
            )
        from ._execution_collective_ffi import collective_schedule

        schedule = collective_schedule(traced.jaxpr, lease_key=lease_key)
        lease._bridge.agree_program(lease._native_id, schedule)
        executable = traced.lower().compile()
        if type(executable) is not Compiled:
            raise TypeError(
                "collective graph lowering did not produce an actual JAX Compiled executable"
            )
        for sharding in jax.tree.leaves(executable.input_shardings):
            lease._require_program_sharding(sharding)
        for sharding in jax.tree.leaves(executable.output_shardings):
            lease._require_program_sharding(sharding)
        object.__setattr__(self, "spec", lease_key)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "executable", executable)
        object.__setattr__(
            self,
            "provenance",
            CollectiveProgramProvenance(jax.__version__, _COLLECTIVE_PRIVATE_DISPATCH),
        )

    def _dispatch(self, *args: Any) -> Result:
        _require_collective_jax_abi()
        # JAX 0.12's AOT C++ cache admits ordered-only programs but omits their
        # runtime tokens. Its Python Compiled.call path retains the same exact
        # executable, PyTree/HiJAX checks, constants, and token input/results.
        # Keep this private JAX parameter access inside the owned artifact.
        result, _, _ = Compiled.call(self.executable._params, *args)
        return result


def prepare_collective_program[Result](
    executable: Callable[..., Result],
    *example_args: Any,
    lease_key: CollectiveLeaseSpec,
) -> CollectiveProgram[Result]:
    """Stage Python once, then own the schedule and compilation of that graph."""
    _require_host_phase("program preparation")
    if isinstance(executable, (Compiled, CollectiveProgram)):
        raise TypeError(
            "prepare_collective_program requires a traceable function, not an opaque compiled artifact"
        )
    lease = require_collective_lease(lease_key)
    lease._admit_arguments(example_args)
    try:
        _require_collective_jax_abi()
        leaves, tree = jax.tree.flatten(example_args)
        shardings: list[Sharding] = []
        for value in leaves:
            if not isinstance(value, Array):
                raise ValueError(
                    "collective program staging requires JAX arrays with explicit placement"
                )
            shardings.append(lease._require_program_sharding(value.sharding))
        input_shardings = tree.unflatten(shardings)
        traced = jax.jit(executable, in_shardings=input_shardings).trace(*example_args)
        program: CollectiveProgram[Result] = CollectiveProgram(
            traced, lease_key=lease_key
        )
        compiled_args, _ = program.executable.input_shardings
        actual_shardings = tree.flatten_up_to(compiled_args)
        for expected, actual in zip(shardings, actual_shardings, strict=True):
            # Public Compiled reports None only for dead, unconsumed arguments.
            if actual is not None and actual != expected:
                raise ValueError(
                    "compiled collective input placement differs from its explicit staging route"
                )
        return program
    except BaseException as error:
        lease._coupled_failure("program preparation", error)
        raise


class CollectiveLease:
    """One live, thread-owned, explicitly retired collective preparation."""

    def __init__(
        self,
        spec: CollectiveLeaseSpec,
        native_id: int,
        bridge: _NativeBridge,
        group: ExecutionGroup,
        source_communicator: Any,
        supervisor: CollectiveSupervisor,
        timeout_seconds: float,
        provenance: CollectiveMPIProvenance,
    ) -> None:
        self.spec = spec
        self.provenance = provenance
        self._native_id = native_id
        self._bridge = bridge
        self._group = group
        self._source_communicator = source_communicator
        self._supervisor = supervisor
        self._timeout_seconds = timeout_seconds
        self._owner = threading.current_thread()
        self._closed = False
        self._failed = False
        self._active = False
        local = tuple(
            device
            for device in group.devices
            if device.process_index == jax.process_index()
        )
        if len(local) != 1:
            raise ValueError("collective lease requires exactly one local group GPU")
        self._local_device = local[0]
        self.binding = CollectiveBinding(
            self._binding_scalar(native_id), self._binding_scalar(0), spec
        )

    def _require_program_sharding(
        self, sharding: Sharding
    ) -> NamedSharding | SingleDeviceSharding:
        if isinstance(sharding, SingleDeviceSharding):
            if sharding.device_set != {self._local_device}:
                raise ValueError(
                    "collective program placement must use the exact local group GPU"
                )
            return sharding
        if isinstance(sharding, NamedSharding):
            keys = tuple(
                (device.process_index, device.id) for device in sharding.mesh.devices.flat
            )
            if keys != self.spec.device_keys:
                raise ValueError(
                    "collective program mesh membership/order does not match its execution group"
                )
            return sharding
        raise TypeError(
            "collective program placement requires SingleDeviceSharding or an exact group NamedSharding"
        )

    def _require_sharding(
        self, sharding: Sharding
    ) -> NamedSharding | SingleDeviceSharding:
        route = self._require_program_sharding(sharding)
        if isinstance(route, NamedSharding) and route.spec != PartitionSpec():
            raise ValueError("global collective scalars must be explicitly replicated")
        return route

    def _binding_scalar(self, value: int, sharding: Sharding | None = None) -> Array:
        host = np.asarray(value, dtype=np.int64)
        if sharding is None:
            return jax.device_put(host, self._local_device)
        route = self._require_sharding(sharding)
        if isinstance(route, SingleDeviceSharding):
            return jax.device_put(host, route)
        # Native preflight proves uniform values; each process supplies its local
        # replica without addressing another process's GPU.
        return jax.make_array_from_process_local_data(route, host, global_shape=())

    def binding_for(self, sharding: Sharding) -> CollectiveBinding:
        """Bind inactive metadata to an explicitly admitted numerical array route."""
        _require_host_phase("binding placement")
        self._require()
        route = self._require_sharding(sharding)
        return CollectiveBinding(
            self._binding_scalar(self._native_id, route),
            self._binding_scalar(0, route),
            self.spec,
        )

    def _require(self) -> None:
        if self._closed:
            raise CollectiveLeaseError("collective lease is closed")
        if self._failed:
            raise CollectiveLeaseError("collective lease has failed")
        if threading.current_thread() is not self._owner:
            raise CollectiveLeaseError(
                "collective lease calls require the owning host thread"
            )
        if jax.process_index() not in self.spec.process_indices:
            raise CollectiveLeaseError(
                "local process is not a collective lease participant"
            )
        self._supervisor.require_active(self.spec.process_indices, self._timeout_seconds)
        self._bridge.validate(self._native_id)

    def _admit_arguments(self, args: tuple[Any, ...]) -> None:
        self._require()
        if self._active:
            raise CollectiveLeaseError(
                "collective invocation cannot be nested or concurrent"
            )
        bindings = jax.tree.leaves(
            args, is_leaf=lambda value: isinstance(value, CollectiveBinding)
        )
        matching = tuple(
            value for value in bindings if isinstance(value, CollectiveBinding)
        )
        if not matching:
            raise ValueError(
                "collective invoke requires an explicit CollectiveBinding in its arguments"
            )
        if any(value.spec != self.spec for value in matching):
            raise ValueError(
                "collective invocation binding configuration or membership does not match its lease"
            )
        for value in matching:
            self._require_sharding(value.lease_id.sharding)
            if value.invocation_ticket.sharding != value.lease_id.sharding:
                raise ValueError(
                    "collective lease ID and invocation ticket must share one admitted placement"
                )

    def _coupled_failure(self, phase: str, error: BaseException) -> None:
        self._failed = True
        reason = f"coupled collective {phase} failed: {error}"
        try:
            self._bridge.fail(self._native_id, reason)
        finally:
            self._supervisor.fail(reason)

    def invoke[Result](self, program: CollectiveProgram[Result], *args: Any) -> Result:
        """Dispatch only the compiled graph whose actual schedule is owned."""
        _require_host_phase("invocation")
        if type(program) is not CollectiveProgram:
            raise TypeError("collective invocation requires an owned CollectiveProgram")
        if program.spec != self.spec:
            raise ValueError(
                "collective program configuration does not match its live lease"
            )
        _require_collective_jax_abi()
        self._admit_arguments(args)
        return self._execute(program._dispatch, args, program.schedule)

    def prepare[Result](
        self,
        preparation: Callable[..., Result],
        *args: Any,
        target_name: str,
        operation_id: str,
    ) -> Result:
        """Admit one owner-declared host descriptor/preparation operation."""
        _require_host_phase("preparation")
        self._admit_arguments(args)
        from ._execution_collective_ffi import collective_call_identity

        schedule = (collective_call_identity(target_name, self.spec, operation_id),)
        return self._execute(preparation, args, schedule)

    def _execute[Result](
        self,
        executable: Callable[..., Result],
        args: tuple[Any, ...],
        schedule: tuple[str, ...],
    ) -> Result:
        self._active = True
        try:
            ticket = self._bridge.begin(self._native_id, schedule)
            rebound: dict[Sharding, CollectiveBinding] = {}

            def rebind(value: Any) -> Any:
                if not isinstance(value, CollectiveBinding):
                    return value
                route = value.lease_id.sharding
                if route not in rebound:
                    rebound[route] = CollectiveBinding(
                        self._binding_scalar(self._native_id, route),
                        self._binding_scalar(ticket, route),
                        self.spec,
                    )
                return rebound[route]

            bound = jax.tree.map(
                rebind, args, is_leaf=lambda value: isinstance(value, CollectiveBinding)
            )
            result = executable(*bound)
            jax.block_until_ready(result)
            jax.effects_barrier()
            self._bridge.end(self._native_id, ticket)
            return result
        except BaseException as error:
            self._coupled_failure("invocation", error)
            raise
        finally:
            self._active = False

    def close(self) -> None:
        """Collectively retire vendor attachments, then both MPI communicators."""
        _require_host_phase("retirement")
        self._require()
        if self._active:
            raise CollectiveLeaseError("cannot close an active collective invocation")
        try:
            jax.effects_barrier()
            self._bridge.retire(self._native_id)
        except BaseException as error:
            self._coupled_failure("retirement", error)
            raise
        self._closed = True
        with _LIVE_LOCK:
            if _LIVE.get(self.spec) is self:
                del _LIVE[self.spec]


def require_collective_lease(spec: CollectiveLeaseSpec) -> CollectiveLease:
    """Validate live local ownership; native handlers separately validate tickets."""
    with _LIVE_LOCK:
        lease = _LIVE.get(spec)
    if lease is None:
        raise CollectiveLeaseError(
            "collective lease is absent, closed, stale, or not a local participant"
        )
    lease._require()
    return lease


def _initialized_mpi_handle(communicator: Any) -> int:
    # Importing mpi4py.MPI can initialize a different world. Only bootstrap may
    # perform that action; inspect its existing canonical module instead.
    mpi: Any = sys.modules.get("mpi4py.MPI")
    if mpi is None:
        raise CollectiveLeaseError(
            "bootstrap must initialize site-compatible mpi4py.MPI before collective admission"
        )
    if not isinstance(communicator, mpi.Comm):
        raise TypeError("source communicator must be an initialized mpi4py MPI.Comm")
    if not mpi.Is_initialized() or mpi.Is_finalized():
        raise CollectiveLeaseError(
            "source MPI runtime must already be initialized and not finalized"
        )
    handle = communicator.py2f()
    if type(handle) is not int:
        raise TypeError(
            "initialized source communicator py2f() must return an integer handle"
        )
    return handle


def create_collective_lease(
    group: ExecutionGroup,
    *,
    provider_id: str,
    configuration_id: str,
    preparation_id: str,
    source_communicator: Any,
    process_rank_map: Mapping[int, int],
    supervisor: CollectiveSupervisor,
    failure_timeout_seconds: float,
) -> CollectiveLease:
    """Bind an initialized group to isolated site-MPI control/numeric contexts.

    ``source_communicator`` is an already-initialized bootstrap communicator,
    exposing mpi4py's ``py2f`` boundary. Its process/rank mapping is explicit and
    independently agreed using actual JAX process indices by native creation.
    """
    _require_host_phase("creation")
    if not group.is_member:
        raise CollectiveLeaseError("local process is not an execution group participant")
    if not math.isfinite(failure_timeout_seconds) or failure_timeout_seconds <= 0:
        raise ValueError("collective failure deadline must be finite and positive")
    if not _jax_config.x64_enabled():
        raise ValueError(
            "collective lease and invocation tickets require JAX int64 support"
        )
    if any(device_platform(device) != "cuda" for device in group.devices):
        raise ValueError(
            "site MPI numerical collective consumers require CUDA execution group devices"
        )
    supervisor.require_active(group.spec.process_indices, failure_timeout_seconds)
    _identity(supervisor.supervisor_id, "supervisor_id")
    source_fortran = _initialized_mpi_handle(source_communicator)
    bridge = _NativeBridge()
    provenance = bridge.provenance()
    spec = CollectiveLeaseSpec(
        group.spec.group_id,
        provider_id,
        configuration_id,
        preparation_id,
        group.spec.process_indices,
        group.spec.device_keys,
        provenance.build_identity,
    )
    spec.require_group(group.spec)
    if set(process_rank_map) != set(spec.process_indices):
        raise ValueError(
            "MPI rank mapping must cover exactly the execution group participants"
        )
    ranks = tuple(process_rank_map[process] for process in spec.process_indices)
    if any(
        type(rank) is not int or rank < 0 or rank > np.iinfo(np.int32).max
        for rank in ranks
    ):
        raise ValueError("source MPI ranks must be nonnegative int32 integers")
    if len(set(ranks)) != len(ranks):
        raise ValueError("source MPI rank mapping must be unique")
    agreement_id = canonical_fingerprint(
        {
            "spec": spec.identity,
            "source_ranks": ranks,
            "supervisor_id": supervisor.supervisor_id,
            "failure_timeout_seconds": failure_timeout_seconds,
            "mpi_library_sha256": provenance.library_sha256,
            "mpi_header_sha256": provenance.header_sha256,
            "mpi_library_version": provenance.library_version,
        }
    )
    with _LIVE_LOCK:
        if spec in _LIVE:
            raise CollectiveLeaseError(
                "this collective preparation already has a live local lease"
            )
        native_id = bridge.create(
            spec,
            source_fortran,
            jax.process_index(),
            ranks,
            agreement_id,
            failure_timeout_seconds,
        )
        try:
            lease = CollectiveLease(
                spec,
                native_id,
                bridge,
                group,
                source_communicator,
                supervisor,
                failure_timeout_seconds,
                provenance,
            )
        except BaseException as error:
            reason = f"coupled collective lease binding failed: {error}"
            try:
                bridge.fail(native_id, reason)
            finally:
                supervisor.fail(reason)
            raise
        _LIVE[spec] = lease
    return lease
