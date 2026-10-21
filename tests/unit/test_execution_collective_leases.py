#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host admission/lifetime contracts; no numerical MPI operation is simulated."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import jax
import numpy as np
import pytest
from jax import Array, make_jaxpr
from jax.sharding import (
    Mesh,
    NamedSharding,
    PartitionSpec,
    Sharding,
    SingleDeviceSharding,
)

import phydrax._execution_collective_ffi as collective_ffi
import phydrax._execution_collectives as collectives
from phydrax import config as pxconfig
from phydrax._execution_collectives import (
    CollectiveBinding,
    CollectiveLease,
    CollectiveLeaseError,
    CollectiveLeaseSpec,
    CollectiveMPIProvenance,
    CollectiveProgram,
    create_collective_lease,
    prepare_collective_program,
    require_collective_lease,
)
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup


@dataclass(frozen=True)
class _Device:
    process_index: int
    id: int


class _Source:
    def py2f(self) -> int:
        return 17


class _Supervisor:
    supervisor_id = "authenticated-launch"

    def __init__(self) -> None:
        self.active = True
        self.failures: list[str] = []
        self.requirements: list[tuple[tuple[int, ...], float]] = []

    def require_active(
        self, process_indices: tuple[int, ...], timeout_seconds: float
    ) -> None:
        self.requirements.append((process_indices, timeout_seconds))
        if not self.active:
            raise RuntimeError("bounded coupled supervisor is not live")

    def fail(self, reason: str) -> None:
        self.failures.append(reason)


class _Bridge:
    """Host ABI boundary double, never an implementation of MPI collectives."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.active_ticket = 0
        self.next_ticket = 0
        self.closed = False
        self.failed = False
        self.create_arguments: tuple[object, ...] = ()
        self.retirement_error = False
        self.expected_schedule: tuple[str, ...] | None = None
        self.admitted_schedule: tuple[str, ...] = ()

    def provenance(self) -> CollectiveMPIProvenance:
        return CollectiveMPIProvenance(
            "site-build", "/site/libmpi.so", "library-digest", "header-digest", "site MPI"
        )

    def create(
        self,
        spec: CollectiveLeaseSpec,
        source_fortran: int,
        process_index: int,
        source_ranks: tuple[int, ...],
        agreement_id: str,
        timeout_seconds: float,
    ) -> int:
        self.create_arguments = (
            spec,
            source_fortran,
            process_index,
            source_ranks,
            agreement_id,
            timeout_seconds,
        )
        self.events.append("create")
        return 71

    def validate(self, lease_id: int) -> None:
        if lease_id != 71 or self.closed or self.failed:
            raise CollectiveLeaseError("native lease is closed, stale, or failed")

    def agree_program(self, lease_id: int, schedule: tuple[str, ...]) -> None:
        self.validate(lease_id)
        self.events.append("agree-program")
        if self.expected_schedule is not None and self.expected_schedule != schedule:
            raise CollectiveLeaseError(
                "all-rank transformed collective schedule disagrees"
            )

    def begin(self, lease_id: int, schedule: tuple[str, ...]) -> int:
        self.validate(lease_id)
        if self.active_ticket:
            raise CollectiveLeaseError("native ticket already active")
        self.events.append("begin")
        if self.expected_schedule is not None and self.expected_schedule != schedule:
            raise CollectiveLeaseError(
                "all-rank transformed collective schedule disagrees"
            )
        self.admitted_schedule = schedule
        self.next_ticket += 1
        self.active_ticket = self.next_ticket
        return self.active_ticket

    def end(self, lease_id: int, ticket: int) -> None:
        self.validate(lease_id)
        if ticket != self.active_ticket or not ticket:
            raise CollectiveLeaseError("native ticket is stale")
        self.events.append("end")
        self.active_ticket = 0

    def fail(self, lease_id: int, reason: str) -> None:
        self.failed = True
        self.events.append("fail")

    def retire(self, lease_id: int) -> None:
        self.validate(lease_id)
        if self.retirement_error:
            raise CollectiveLeaseError("all-rank retirement preparation disagrees")
        self.events.append("retire")
        self.closed = True


@dataclass
class _Harness:
    group: ExecutionGroup
    bridge: _Bridge
    supervisor: _Supervisor

    def create(self, **overrides: Any) -> CollectiveLease:
        options: dict[str, Any] = {
            "provider_id": "cudecomp-site",
            "configuration_id": "configuration",
            "preparation_id": "slab-layout",
            "source_communicator": _Source(),
            "process_rank_map": {0: 2},
            "supervisor": self.supervisor,
            "failure_timeout_seconds": 20.0,
        }
        options.update(overrides)
        return create_collective_lease(self.group, **options)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    local = jax.local_devices()[0]
    group = ExecutionGroup(ExecutionGroupSpec("group", (0,), ((0, local.id),)), (local,))
    bridge = _Bridge()
    supervisor = _Supervisor()
    monkeypatch.setattr(collectives, "_NativeBridge", lambda: bridge)
    monkeypatch.setattr(collectives, "_LIVE", {})
    monkeypatch.setattr(collectives.jax, "process_index", lambda: 0)
    monkeypatch.setattr(collectives, "device_platform", lambda device: "cuda")
    monkeypatch.setattr(
        collectives, "_initialized_mpi_handle", lambda source: source.py2f()
    )

    def local_scalar(
        lease: CollectiveLease, value: int, sharding: Sharding | None = None
    ) -> Array:
        return jax.device_put(np.asarray(value, dtype=np.int64), sharding)

    monkeypatch.setattr(CollectiveLease, "_binding_scalar", local_scalar)
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield _Harness(group, bridge, supervisor)


def _ticket_program(lease: CollectiveLease) -> CollectiveProgram[Array]:
    def ticket(binding: CollectiveBinding) -> Array:
        return binding.invocation_ticket

    return prepare_collective_program(ticket, lease.binding, lease_key=lease.spec)


def _spec() -> CollectiveLeaseSpec:
    return CollectiveLeaseSpec(
        "group",
        "provider",
        "configuration",
        "preparation",
        (0, 1),
        ((0, 4), (1, 8)),
        "site-build",
    )


def test_collective_spec_binds_membership_configuration_and_site_build() -> None:
    spec = _spec()
    assert hash(spec) == hash(replace(spec))
    assert spec.identity == replace(spec).identity
    for modified in (
        replace(spec, configuration_id="other"),
        replace(spec, preparation_id="other"),
        replace(spec, provider_id="other"),
        replace(spec, mpi_build_identity="other"),
    ):
        assert modified.identity != spec.identity
    spec.require_group(ExecutionGroupSpec("group", (0, 1), ((0, 4), (1, 8))))
    with pytest.raises(ValueError, match="membership"):
        spec.require_group(ExecutionGroupSpec("other", (0, 1), ((0, 4), (1, 8))))


@pytest.mark.parametrize("devices", [((0, 4), (0, 5)), ((1, 8), (0, 4)), ((0, 4),)])
def test_collective_spec_refuses_wrong_one_gpu_per_rank_membership(
    devices: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError, match="one.GPU|device order"):
        replace(_spec(), device_keys=devices)


def test_create_uses_explicit_initialized_source_and_rank_mapping(
    harness: _Harness,
) -> None:
    lease = harness.create()
    assert harness.bridge.create_arguments[1:4] == (17, 0, (2,))
    assert require_collective_lease(lease.spec) is lease
    assert lease.binding.group_id == "group"
    assert lease.binding.preparation_id == "slab-layout"
    assert len(jax.tree.leaves(lease.binding)) == 2
    assert lease.binding.lease_id.dtype == np.dtype(np.int64)
    assert lease.binding.invocation_ticket.shape == ()
    assert int(lease.binding.invocation_ticket) == 0
    lease.close()


def test_nonparticipant_is_refused_before_native_creation(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collectives.jax, "process_index", lambda: 1)
    with pytest.raises(CollectiveLeaseError, match="participant"):
        harness.create()
    assert not harness.bridge.events


@pytest.mark.parametrize("mapping", [{}, {0: 0, 1: 0}, {0: 0, 1: 1, 2: 2}])
def test_incomplete_or_extra_mpi_rank_mapping_is_refused(
    harness: _Harness, mapping: dict[int, int]
) -> None:
    with pytest.raises(ValueError, match="mapping"):
        harness.create(process_rank_map=mapping)
    assert not harness.bridge.events


def test_supervisor_loss_refuses_admission_before_mpi(harness: _Harness) -> None:
    harness.supervisor.active = False
    with pytest.raises(RuntimeError, match="supervisor"):
        harness.create()
    assert not harness.bridge.events


def test_duplicate_live_preparation_is_refused(harness: _Harness) -> None:
    lease = harness.create()
    with pytest.raises(CollectiveLeaseError, match="already"):
        harness.create()
    assert harness.bridge.events == ["create"]
    lease.close()


def test_invoke_rebinds_dynamic_tickets_and_drains_effects(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = harness.create()
    program = _ticket_program(lease)

    def arrays_ready(value: Any) -> Any:
        harness.bridge.events.append("arrays-ready")
        return value

    def effects_ready() -> None:
        harness.bridge.events.append("effects-ready")

    monkeypatch.setattr(collectives.jax, "block_until_ready", arrays_ready)
    monkeypatch.setattr(collectives.jax, "effects_barrier", effects_ready)
    original = lease.binding
    observed = [
        int(lease.invoke(program, original)),
        int(lease.invoke(program, original)),
    ]
    assert observed == [1, 2]
    assert int(original.invocation_ticket) == 0
    assert harness.bridge.events == [
        "create",
        "agree-program",
        "begin",
        "arrays-ready",
        "effects-ready",
        "end",
        "begin",
        "arrays-ready",
        "effects-ready",
        "end",
    ]
    assert harness.bridge.active_ticket == 0
    lease.close()


def test_binding_configuration_mismatch_is_refused_before_ticket(
    harness: _Harness,
) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    wrong = CollectiveBinding(
        lease.binding.lease_id,
        lease.binding.invocation_ticket,
        replace(lease.spec, configuration_id="other"),
    )
    with pytest.raises(ValueError, match="configuration or membership"):
        lease.invoke(program, wrong)
    with pytest.raises(CollectiveLeaseError, match="absent"):
        require_collective_lease(wrong.spec)
    assert harness.bridge.events == ["create", "agree-program"]
    lease.close()


def test_cross_thread_invocation_is_refused_before_dispatch(harness: _Harness) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lease.invoke, program, lease.binding)
        with pytest.raises(CollectiveLeaseError, match="owning host thread"):
            future.result()
    assert harness.bridge.events == ["create", "agree-program"]
    lease.close()


def test_nested_invocation_is_refused_without_replacing_active_ticket(
    harness: _Harness,
) -> None:
    lease = harness.create()
    program = _ticket_program(lease)

    def executable(binding: CollectiveBinding) -> None:
        with pytest.raises(CollectiveLeaseError, match="nested or concurrent"):
            lease.invoke(program, binding)
        assert int(binding.invocation_ticket) == harness.bridge.active_ticket

    lease.prepare(
        executable,
        lease.binding,
        target_name="descriptor-prepare",
        operation_id="nested-refusal",
    )
    assert harness.bridge.events == ["create", "agree-program", "begin", "end"]
    lease.close()


def test_host_preparation_failure_propagates_to_supervisor_and_closes_admission(
    harness: _Harness,
) -> None:
    lease = harness.create()

    def fail(binding: CollectiveBinding) -> None:
        raise ValueError("rank-local preparation failed")

    with pytest.raises(ValueError, match="rank-local preparation"):
        lease.prepare(
            fail,
            lease.binding,
            target_name="descriptor-prepare",
            operation_id="rank-local-failure",
        )
    assert harness.bridge.events == ["create", "begin", "fail"]
    assert harness.supervisor.failures == [
        "coupled collective invocation failed: rank-local preparation failed"
    ]
    with pytest.raises(CollectiveLeaseError, match="failed"):
        require_collective_lease(lease.spec)
    with pytest.raises(CollectiveLeaseError, match="failed"):
        lease.close()


def test_retirement_is_explicit_and_closed_reuse_is_refused(harness: _Harness) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    lease.close()
    assert harness.bridge.events == ["create", "agree-program", "retire"]
    with pytest.raises(CollectiveLeaseError, match="closed"):
        lease.invoke(program, lease.binding)
    with pytest.raises(CollectiveLeaseError, match="closed"):
        lease.close()
    with pytest.raises(CollectiveLeaseError, match="closed"):
        require_collective_lease(lease.spec)


def test_retirement_disagreement_uses_failure_channel_not_status_collective(
    harness: _Harness,
) -> None:
    lease = harness.create()
    harness.bridge.retirement_error = True
    with pytest.raises(CollectiveLeaseError, match="all-rank retirement"):
        lease.close()
    assert harness.bridge.events == ["create", "fail"]
    assert len(harness.supervisor.failures) == 1


def test_asymmetric_transformed_activity_refuses_before_any_execution(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = harness.create()
    primal = collective_ffi.collective_call_identity("solve", lease.spec, "primal")
    tangent = collective_ffi.collective_call_identity("solve", lease.spec, "tangent")
    harness.bridge.expected_schedule = (primal, tangent)
    monkeypatch.setattr(
        collective_ffi, "collective_schedule", lambda program, *, lease_key: (primal,)
    )
    executed: list[bool] = []

    def executable(binding: CollectiveBinding) -> None:
        executed.append(True)

    with pytest.raises(
        CollectiveLeaseError, match="transformed collective schedule disagrees"
    ):
        prepare_collective_program(executable, lease.binding, lease_key=lease.spec)
    assert executed == [True]
    assert harness.bridge.events == ["create", "agree-program", "fail"]
    assert harness.bridge.active_ticket == 0
    assert len(harness.supervisor.failures) == 1


def test_unadmitted_transformed_program_uses_coupled_failure_channel(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = harness.create()

    def reject_schedule(
        program: Any, *, lease_key: CollectiveLeaseSpec
    ) -> tuple[str, ...]:
        raise ValueError("collective activity depends on dynamic control flow")

    monkeypatch.setattr(collective_ffi, "collective_schedule", reject_schedule)
    with pytest.raises(ValueError, match="dynamic control flow"):
        prepare_collective_program(
            lambda binding: None, lease.binding, lease_key=lease.spec
        )
    assert harness.bridge.events == ["create", "fail"]
    assert "program preparation" in harness.supervisor.failures[0]


def test_host_preparation_has_separate_owner_generated_admission(
    harness: _Harness,
) -> None:
    lease = harness.create()
    observed: list[int] = []

    def descriptor_preparation(binding: CollectiveBinding) -> None:
        observed.append(int(binding.invocation_ticket))
        harness.bridge.events.append("descriptor-preparation")

    lease.prepare(
        descriptor_preparation,
        lease.binding,
        target_name="phydrax_cudecomp_prepare",
        operation_id="canonical-descriptor",
    )
    assert harness.bridge.admitted_schedule == (
        collective_ffi.collective_call_identity(
            "phydrax_cudecomp_prepare", lease.spec, "canonical-descriptor"
        ),
    )
    assert observed == [1]
    assert harness.bridge.events == ["create", "begin", "descriptor-preparation", "end"]
    lease.close()


def test_supervisor_must_remain_live_at_each_invocation(harness: _Harness) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    harness.supervisor.active = False
    with pytest.raises(RuntimeError, match="supervisor"):
        lease.invoke(program, lease.binding)
    assert harness.bridge.events == ["create", "agree-program"]


def test_missing_initialized_mpi_boundary_is_not_imported_or_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    with pytest.raises(CollectiveLeaseError, match="bootstrap must initialize"):
        collectives._initialized_mpi_handle(_Source())
    assert "mpi4py.MPI" not in sys.modules


def test_binding_defaults_to_rank_local_gpu_placement(harness: _Harness) -> None:
    lease = harness.create()
    assert isinstance(lease.binding.lease_id.sharding, SingleDeviceSharding)
    assert lease.binding.lease_id.sharding == lease.binding.invocation_ticket.sharding
    lease.close()


def test_explicit_global_binding_preserves_route_when_ticket_is_rebound(
    harness: _Harness,
) -> None:
    local = jax.local_devices()[0]
    harness.group = ExecutionGroup(
        ExecutionGroupSpec("one-rank", (0,), ((0, local.id),)), (local,)
    )
    lease = harness.create(process_rank_map={0: 0})
    mesh = Mesh(np.asarray([local], dtype=object), ("device",))
    replicated = NamedSharding(mesh, PartitionSpec())
    global_binding = lease.binding_for(replicated)
    observed: list[tuple[Sharding, Sharding]] = []

    def executable(
        local_binding: CollectiveBinding, shared_binding: CollectiveBinding
    ) -> None:
        observed.append(
            (local_binding.lease_id.sharding, shared_binding.lease_id.sharding)
        )
        assert local_binding.invocation_ticket.sharding == local_binding.lease_id.sharding
        assert shared_binding.invocation_ticket.sharding == replicated
        assert (
            int(local_binding.invocation_ticket)
            == int(shared_binding.invocation_ticket)
            == 1
        )

    lease.prepare(
        executable,
        lease.binding,
        global_binding,
        target_name="descriptor-prepare",
        operation_id="placement",
    )
    assert observed == [(lease.binding.lease_id.sharding, replicated)]
    lease.close()


def test_global_binding_refuses_foreign_group_mesh(harness: _Harness) -> None:
    lease = harness.create()
    local = jax.local_devices()[0]
    mesh = Mesh(np.asarray([local, local], dtype=object), ("device",))
    with pytest.raises(ValueError, match="mesh membership/order"):
        lease.binding_for(NamedSharding(mesh, PartitionSpec()))
    assert harness.bridge.events == ["create"]
    lease.close()


def test_global_metadata_binding_must_be_replicated(harness: _Harness) -> None:
    lease = harness.create()
    mesh = Mesh(np.asarray([jax.local_devices()[0]], dtype=object), ("device",))
    with pytest.raises(ValueError, match="explicitly replicated"):
        lease.binding_for(NamedSharding(mesh, PartitionSpec("device")))
    lease.close()


def test_local_metadata_binding_refuses_another_device_owner(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = harness.create()
    requested = lease.binding.lease_id.sharding
    monkeypatch.setattr(lease, "_local_device", _Device(0, 999))
    with pytest.raises(ValueError, match="exact local group GPU"):
        lease.binding_for(requested)
    lease.close()


def test_metadata_leaves_cannot_mix_local_and_global_array_routes(
    harness: _Harness,
) -> None:
    local = jax.local_devices()[0]
    harness.group = ExecutionGroup(
        ExecutionGroupSpec("one-rank", (0,), ((0, local.id),)), (local,)
    )
    lease = harness.create(process_rank_map={0: 0})
    mesh = Mesh(np.asarray([local], dtype=object), ("device",))
    global_binding = lease.binding_for(NamedSharding(mesh, PartitionSpec()))
    mixed = CollectiveBinding(
        lease.binding.lease_id, global_binding.invocation_ticket, lease.spec
    )
    with pytest.raises(ValueError, match="share one admitted placement"):
        lease.prepare(
            lambda binding: None,
            mixed,
            target_name="descriptor-prepare",
            operation_id="mixed-placement",
        )
    assert harness.bridge.events == ["create"]
    lease.close()


@pytest.mark.parametrize(
    "operation", ["binding", "prepare", "program", "invoke", "close", "create"]
)
def test_host_phases_refuse_tracing_even_with_captured_real_bindings(
    harness: _Harness,
    operation: str,
) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    requirements = len(harness.supervisor.requirements)

    def traced_host_operation() -> None:
        match operation:
            case "binding":
                lease.binding_for(lease.binding.lease_id.sharding)
            case "prepare":
                lease.prepare(
                    lambda binding: None,
                    lease.binding,
                    target_name="descriptor-prepare",
                    operation_id="recipe",
                )
            case "invoke":
                lease.invoke(program, lease.binding)
            case "close":
                lease.close()
            case "program":
                prepare_collective_program(
                    lambda binding: None, lease.binding, lease_key=lease.spec
                )
            case "create":
                harness.create()
            case _:
                raise AssertionError("unexpected test operation")

    with pytest.raises(ValueError, match="host phase.*JAX transformation"):
        make_jaxpr(traced_host_operation)()
    assert harness.bridge.events == ["create", "agree-program"]
    assert len(harness.supervisor.requirements) == requirements
    lease.close()


@pytest.mark.parametrize("example_kind", ["numpy", "abstract"])
def test_program_staging_refuses_unknown_numeric_placement_before_tracing(
    harness: _Harness,
    example_kind: str,
) -> None:
    lease = harness.create()
    value = (
        np.ones((2,), dtype=np.float64)
        if example_kind == "numpy"
        else jax.ShapeDtypeStruct((2,), np.dtype(np.float64))
    )
    staged: list[bool] = []

    def executable(binding: CollectiveBinding, values: Any) -> Array:
        staged.append(True)
        return binding.invocation_ticket

    with pytest.raises(ValueError, match="explicit placement"):
        prepare_collective_program(executable, lease.binding, value, lease_key=lease.spec)
    assert staged == []
    assert harness.bridge.events == ["create", "fail"]
    assert harness.bridge.active_ticket == 0
    assert len(harness.supervisor.failures) == 1


@pytest.mark.parametrize("release", ["0.12.1", "0.12.0.dev0"])
def test_unqualified_jax_dispatch_abi_is_refused_before_staging(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    release: str,
) -> None:
    lease = harness.create()
    staged: list[bool] = []

    def executable(binding: CollectiveBinding) -> Array:
        staged.append(True)
        return binding.invocation_ticket

    monkeypatch.setattr(collectives.jax, "__version__", release)
    with pytest.raises(
        CollectiveLeaseError, match="qualified JAX 0.12.0.*unqualified JAX"
    ):
        prepare_collective_program(executable, lease.binding, lease_key=lease.spec)
    assert staged == []
    assert harness.bridge.events == ["create", "fail"]
    assert harness.bridge.active_ticket == 0


def test_program_records_qualified_private_dispatch_provenance_and_refuses_changed_abi(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = harness.create()
    program = _ticket_program(lease)
    assert program.provenance.jax_release == "0.12.0"
    assert (
        "Compiled.call/Compiled._params" in program.provenance.private_dispatch_identity
    )
    assert (
        program.provenance.identity
        != replace(program.provenance, jax_release="unqualified").identity
    )
    monkeypatch.setattr(collectives.jax, "__version__", "0.12.1")
    with pytest.raises(CollectiveLeaseError, match="qualified JAX 0.12.0"):
        lease.invoke(program, lease.binding)
    assert harness.bridge.events == ["create", "agree-program"]
    assert harness.bridge.active_ticket == 0
    lease.close()
