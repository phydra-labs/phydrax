# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real site-MPI/JAX worker for the bounded, authenticated MPI launcher.

Launch through ``launch_local_processes(..., mpi_launcher=(site_mpiexec, ...))``.
Independent Python ranks are not an MPI world and are explicitly refused. The
worker owns optional mpi4py bootstrap; the production lease bridge never does.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from jax import Array

    from phydrax._execution_collectives import (
        CollectiveBinding,
        CollectiveLease,
        CollectiveProgram,
    )


# Keep import-time discovery and MPI initialization out of this module.
def _bootstrap() -> tuple[Any, Any, Any]:
    import mpi4py

    mpi4py.rc.initialize = True
    mpi4py.rc.finalize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"
    from mpi4py import MPI

    from phydrax import config
    from phydrax._execution_launch import require_local_process_supervision
    from phydrax._execution_runtime import ExecutionRuntime

    supervisor = require_local_process_supervision()
    source = MPI.COMM_WORLD
    if (
        source.Get_size() != supervisor.num_processes
        or source.Get_rank() != supervisor.process_id
    ):
        raise RuntimeError(
            "actual initialized MPI world does not match the authenticated supervised rank world"
        )
    if source.Get_size() <= 1:
        raise RuntimeError(
            "collective qualification requires a real multi-rank MPI world, not separate singleton jobs"
        )
    if MPI.Query_thread() != MPI.THREAD_MULTIPLE:
        raise RuntimeError("site MPI bootstrap did not provide MPI_THREAD_MULTIPLE")
    config.configure(config.RuntimeConfig(enable_x64=True))
    runtime = ExecutionRuntime.current(mesh_axes=(("spectral_0", source.Get_size()),))
    return source, supervisor, runtime


def _closed_program(lease: CollectiveLease) -> CollectiveProgram[Array]:
    from phydrax._execution_collectives import prepare_collective_program

    def inactive_ticket(binding: CollectiveBinding) -> Array:
        return binding.invocation_ticket

    return prepare_collective_program(
        inactive_ticket, lease.binding, lease_key=lease.spec
    )


def _closed_refusal(
    lease: CollectiveLease,
    program: CollectiveProgram[Array],
) -> dict[str, object]:
    from phydrax._execution_collectives import CollectiveLeaseError

    try:
        lease.invoke(program, lease.binding)
    except CollectiveLeaseError as error:
        if "closed" not in str(error):
            raise AssertionError(
                f"closed lease returned the wrong refusal: {error}"
            ) from error
        return {"closed_reuse_refused": True, "message": str(error)}
    raise AssertionError("closed collective lease was reused")


def _thread_refusal(lease: Any) -> None:
    from phydrax._execution_collectives import (
        CollectiveLeaseError,
        require_collective_lease,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(require_collective_lease, lease.spec)
        try:
            pending.result()
        except CollectiveLeaseError as error:
            if "owning host thread" not in str(error):
                raise AssertionError(
                    f"wrong cross-thread lease refusal: {error}"
                ) from error
            return
    raise AssertionError("collective lease admitted a different host thread")


def run(arguments: argparse.Namespace) -> dict[str, object]:
    import jax
    import numpy as np

    from phydrax._execution_collectives import create_collective_lease
    from phydrax._fingerprint import canonical_fingerprint

    source, supervisor, runtime = _bootstrap()
    group = runtime.root_group
    if (
        jax.process_count() != source.Get_size()
        or jax.process_index() != source.Get_rank()
    ):
        raise RuntimeError(
            "initialized JAX process coordinates differ from actual MPI rank coordinates"
        )
    supervisor.require_active(
        group.spec.process_indices, arguments.failure_timeout_seconds
    )
    actual_processes = tuple(source.allgather(jax.process_index()))
    if len(set(actual_processes)) != source.Get_size():
        raise RuntimeError(
            "MPI source ranks do not identify unique initialized JAX processes"
        )
    process_rank_map = {process: rank for rank, process in enumerate(actual_processes)}
    provider = "cudecomp-site" if arguments.case == "cudecomp" else "cudss-mgmn"
    configuration_id = canonical_fingerprint(
        {
            "provider": provider,
            "dtype": arguments.dtype,
            "processes": group.spec.process_indices,
            "profile": "real-site-mpi-collective-qualification",
        }
    )
    preparation_id = (
        "cudecomp-spectral-qualification"
        if arguments.case == "cudecomp"
        else "cudss-mgmn-fivepoint-globalpattern"
    )
    if (
        arguments.failure_case == "configuration"
        and source.Get_rank() == source.Get_size() - 1
    ):
        configuration_id += "-asymmetric"
    if (
        arguments.failure_case == "preparation"
        and source.Get_rank() == source.Get_size() - 1
    ):
        preparation_id += "-asymmetric"
    records: list[dict[str, object]] = []
    generations: list[int] = []
    stable_identity: str | None = None
    previous_spectral_plan: Any = None
    spectral_identity: tuple[object, object] | None = None
    for generation in range(arguments.rounds):
        lease = create_collective_lease(
            group,
            provider_id=provider,
            configuration_id=configuration_id,
            preparation_id=preparation_id,
            source_communicator=source,
            process_rank_map=process_rank_map,
            supervisor=supervisor,
            failure_timeout_seconds=arguments.failure_timeout_seconds,
        )
        if arguments.failure_case in ("configuration", "preparation"):
            supervisor.fail(
                "asymmetric collective lease preparation unexpectedly succeeded"
            )
            raise AssertionError(
                "asymmetric collective lease preparation was not refused"
            )
        if stable_identity is not None and stable_identity != lease.spec.identity:
            raise AssertionError(
                "explicit lease reconstruction changed the stable preparation identity"
            )
        stable_identity = lease.spec.identity
        native_id = np.asarray(jax.device_get(lease.binding.lease_id)).item()
        if native_id in generations:
            raise AssertionError("retired collective native lease generation was reused")
        generations.append(native_id)
        _thread_refusal(lease)
        closed_program = _closed_program(lease)
        if arguments.failure_case == "closed-lease-invoke":
            lease.close()
            evidence = _closed_refusal(lease, closed_program)
        elif arguments.case == "cudecomp":
            from phydrax.discretization.spectral import SpectralMeshTopology
            from tests._support.cudecomp_spectral_cases import (
                run_cudecomp_cases,
                run_cudecomp_failure_case,
            )

            topology = SpectralMeshTopology(
                group.mesh, execution_group_id=group.spec.group_id
            )
            if arguments.failure_case:
                evidence = run_cudecomp_failure_case(
                    lease, topology, arguments.dtype, arguments.failure_case
                )
            else:
                if previous_spectral_plan is not None:
                    previous_spectral_plan = previous_spectral_plan.prepare()
                evidence, previous_spectral_plan = run_cudecomp_cases(
                    lease, topology, arguments.dtype
                )
                current_identity = (evidence["numerical_id"], evidence["execution_id"])
                if (
                    spectral_identity is not None
                    and spectral_identity != current_identity
                ):
                    raise AssertionError(
                        "lease reconstruction changed spectral scientific or static execution identity"
                    )
                spectral_identity = current_identity
                if arguments.benchmark:
                    from benchmarks.distributed_spectral import _run

                    benchmark = _run(
                        "physical-slab",
                        "slab",
                        shape=(32, 32, 32),
                        mesh_shape=(source.Get_size(),),
                        precision_name=arguments.dtype,
                        repeats=3,
                        maximum_bytes=1 << 31,
                        transpose_provider="cudecomp",
                        collective_lease=lease,
                        maximum_external_scratch_bytes=64 << 20,
                    )
                    evidence["benchmark"] = benchmark
                    if not benchmark["passed"]:
                        print(
                            json.dumps(
                                {
                                    "case": "cudecomp-benchmark-failure",
                                    "rank": jax.process_index(),
                                    "benchmark": benchmark,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        raise ArithmeticError(
                            "cuDecomp benchmark failed its numerical or admission gate"
                        )
            lease.close()
            evidence.update(_closed_refusal(lease, closed_program))
        else:
            from tests.integration.test_cudss_mgmn import (
                run_mgmn_cases,
                run_mgmn_failure_case,
            )

            if arguments.failure_case:
                evidence = run_mgmn_failure_case(
                    lease, source, arguments.dtype, arguments.failure_case
                )
            else:
                evidence = run_mgmn_cases(lease, source, arguments.dtype)
            lease.close()
            evidence.update(_closed_refusal(lease, closed_program))
        records.append(
            {
                "generation": generation,
                "native_lease_id": native_id,
                "lease_spec_id": lease.spec.identity,
                "mpi_provenance": asdict(lease.provenance),
                "program_dispatch_provenance": asdict(closed_program.provenance),
                "owner_thread_refused": True,
                "evidence": evidence,
            }
        )
    result: dict[str, object] = {
        "case": arguments.case,
        "dtype": arguments.dtype,
        "rank": jax.process_index(),
        "process_count": jax.process_count(),
        "group_id": group.spec.group_id,
        "supervisor_id": supervisor.supervisor_id,
        "rounds": records,
    }
    # Every lease and vendor attachment is retired before either runtime owner.
    jax.distributed.shutdown()
    from mpi4py import MPI

    MPI.Finalize()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("cudecomp", "mgmn"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--failure-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--failure-case", default="")
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    if arguments.rounds < 1:
        parser.error("rounds must be positive")
    if arguments.benchmark and (arguments.case != "cudecomp" or arguments.failure_case):
        parser.error("benchmark requires a healthy cuDecomp case")
    try:
        result = run(arguments)
    except BaseException as error:
        from phydrax._execution_launch import require_local_process_supervision

        require_local_process_supervision().fail(f"collective worker failed: {error}")
        raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
