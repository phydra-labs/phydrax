#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Separately observed qualification gates of distributed atomistic domains.

Each gate reports ``PASS`` only when its topology was actually observed in
this invocation and every functional check passed, ``FAIL`` when it was
observed and a check failed, and ``NOT_OBSERVED`` (with the reason) otherwise.
No gate inherits another gate's evidence:

``lane-reference``
    Owner regions as named vmap lanes of one device (halo and graph
    partition, 2 and 3 owners) against the independent single-device global
    MACE oracle. A numerical oracle, never hardware evidence.
``single-process-physical-devices``
    One owner per physical accelerator of ONE process, admitted by the
    existing `resolve_execution_plan` resource admission. Forced or host-CPU
    devices are recorded as functional evidence and reported NOT_OBSERVED.
``multi-process-one-host``
    Two coupled local processes (`launch_local_processes`, finite deadline)
    with process-local ingress, both strategies, all ranks on one observed
    Linux boot identity.
``multi-host``
    At least two observed physical hosts (distinct Linux boot identities in
    the live `ExecutionRuntime` inventory, admitted with ``host_count=2``).
    Observable only with ``--observe-current-job`` on every rank of an
    externally launched multi-host job; a local launch never observes it.
``killed-rank-fail-coupled``
    A SIGKILLed rank fails the coupled job before its deadline and no global
    checkpoint commit exists (domain publication and stage-transport probe).
``same-topology-restart`` / ``changed-topology-restart``
    Distinct checks: bitwise restore on the publishing topology versus a
    tolerance-class continuation on another owner/process topology with
    explicit lineage evidence.
``stage-transport-subgroup``
    The real two-process stage-transport probe (child stage collectives,
    credits, parked packets, durable commit and bitwise restore).
``stage-domain-campaign``
    A backpressured two-stage native domain-dynamics campaign over
    process-subset child groups: credit and sink parking without physics, a
    killed rank during a commit, bitwise same-topology restart and a
    changed-topology migration of the committed campaign.
``data-domain-training``
    Device data × domain training with both mesh axes > 1, unequal and empty
    support, an independent single-device objective/gradient reference and a
    durable restart.
``model-family-domains``
    AIMNet2, UMA, MACE-LES and LJ/PME domain routes against independent
    single-device programs (both strategies) and dynamic-cell MTK
    trajectories; undeclared routes must be natively refused.

Records carry the benchmark source/driver identity, the runtime environment
and the observed process/host topology. Accelerator vendors, specific GPUs,
multi-host transports and pretrained-model fidelity are never assumed.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, Literal

import jax
import numpy as np
from jax.experimental import multihost_utils

from benchmarks._runtime import capture_benchmark_identity, capture_environment
from phydrax import config as pxconfig
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import bind_execution_group
from phydrax._fingerprint import canonical_fingerprint
from phydrax.execution import (
    DistributionMode,
    ExecutionAdmissionError,
    ExecutionCandidate,
    ExecutionPolicy,
    ExecutionRequirements,
    ExecutionRuntime,
    resolve_execution_plan,
    ResourceRequest,
)
from tools._accelerator_process import write_json_evidence
from tools.atomistic_distributed_smoke import (
    build_domain,
    compare_digests,
    json_safe,
    oracle_parity,
    owner_address,
    Platform,
    process_local_ingress,
    run_rank,
    run_scenario,
    runtime_record,
    smoke_structure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY = "atomistic-distributed-execution"
GateStatus = Literal["PASS", "FAIL", "NOT_OBSERVED"]
GATES = (
    "lane-reference",
    "single-process-physical-devices",
    "multi-process-one-host",
    "multi-host",
    "killed-rank-fail-coupled",
    "same-topology-restart",
    "changed-topology-restart",
    "stage-transport-subgroup",
    "stage-domain-campaign",
    "data-domain-training",
    "model-family-domains",
)
STRATEGIES = ("halo", "graph-partition")
NONCLAIMS = (
    "Lane-reference and forced host-CPU devices are functional oracles of the "
    "collective program, not accelerator, scaling or multi-host evidence.",
    "A multi-process launch on one host is not multi-host evidence; multi-host "
    "requires two or more distinct observed Linux boot identities.",
    "No accelerator vendor, GPU model, interconnect or multi-host transport "
    "provider is assumed present or qualified by absence of failure.",
    "Model fidelity is checked against the same randomly initialized native "
    "model on one device; pretrained-model fidelity is a separate gate.",
    "Rank loss is fail-coupled by the launcher: the job terminates and restarts "
    "from its last durable global commit; no collective rollback is promised.",
    "Changed-topology continuation preserves scientific state and RNG lineage "
    "within a declared tolerance; future trajectories are not bitwise identical.",
)


def _gate(
    name: str,
    status: GateStatus,
    reason: str,
    /,
    *,
    scope: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if name not in GATES or status not in get_args(GateStatus):
        raise ValueError(f"Unknown gate {name!r} or status {status!r}.")
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "scope": scope,
        "evidence": {} if evidence is None else dict(evidence),
    }


def _guarded(
    name: str, scope: str, check: Callable[[], dict[str, Any]], /
) -> dict[str, Any]:
    """Run one gate; an exception is an observed failure, never a silent skip."""
    try:
        return check()
    except Exception as error:
        return _gate(
            name,
            "FAIL",
            f"{type(error).__name__}: {error}"[:4000],
            scope=scope,
        )


# ------------------------------------------------------------ in-process


def _device_subset(runtime: ExecutionRuntime, owners: int, /) -> Any:
    """The first ``owners`` devices of the root group as one owner mesh."""
    parent = runtime.root_group.spec
    keys = parent.device_keys[:owners]
    return bind_execution_group(
        ExecutionGroupSpec(
            canonical_fingerprint(
                {
                    "parent_group_id": parent.group_id,
                    "group_index": 0,
                    "device_keys": [list(key) for key in keys],
                }
            ),
            parent.process_indices,
            keys,
            mesh_axes=(("device", owners),),
            parent_group_id=parent.group_id,
            process_host_ids=parent.process_host_ids,
        )
    )


def _ownership_parities(ownership: Any, /) -> dict[str, Any]:
    """Process-local ingress and oracle parity for both spatial strategies."""
    positions, species_ids, momenta = smoke_structure()
    results = {}
    for strategy in STRATEGIES:
        model, runtime = build_domain(ownership, strategy)
        state = process_local_ingress(runtime, model, positions, species_ids, momenta)
        results[strategy] = oracle_parity(runtime, model, state, positions)
    return results


def _lane_reference() -> dict[str, Any]:
    from phydrax.discretization.spatial._distributed_relations import (
        DistributedOwnershipPlan,
    )
    from tests._support.atomistic_domain import LOCAL_CAPACITY

    scope = "numerical oracle on one device; not hardware evidence"
    if jax.process_count() != 1:
        return _gate(
            "lane-reference",
            "NOT_OBSERVED",
            "the lane reference is qualified in a single-process invocation",
            scope=scope,
        )
    runtime = ExecutionRuntime.current()
    group = runtime.child_groups(runtime.root_group.spec.device_count)[0]
    evidence = {}
    for owners in (2, 3):
        ownership = DistributedOwnershipPlan(
            owner_address(), group, LOCAL_CAPACITY, owner_lanes=owners
        )
        evidence[f"{owners}-lanes"] = _ownership_parities(ownership)
    errors = [
        f"{lanes}/{strategy}: {error}"
        for lanes, results in evidence.items()
        for strategy, result in results.items()
        for error in result["errors"]
    ]
    return _gate(
        "lane-reference",
        "FAIL" if errors else "PASS",
        "; ".join(errors) or "lane owners match the single-device oracle",
        scope=scope,
        evidence=evidence,
    )


def _admission(
    runtime: ExecutionRuntime,
    group: ExecutionGroupSpec,
    request: ResourceRequest,
    owner_id: str,
    /,
) -> dict[str, Any]:
    """Existing resource/topology admission of one candidate group."""
    requirements = ExecutionRequirements(owner_id)
    plan = resolve_execution_plan(
        ExecutionPolicy(DistributionMode.DISTRIBUTED, resources=request),
        runtime.inventory,
        (
            ExecutionCandidate(
                owner_id,
                requirements.requirements_id,
                group,
                estimated_memory_bytes=request.memory_bytes,
            ),
        ),
        precision_policy_id="atomistic-domain-float64",
        solver_policy_id="atomistic-domain-qualification",
    )
    return {
        "execution_plan_id": plan.execution_plan_id,
        "request": request.to_payload(),
    }


def _single_process_physical_devices() -> dict[str, Any]:
    from phydrax.discretization.spatial._distributed_relations import (
        DistributedOwnershipPlan,
    )
    from tests._support.atomistic_domain import LOCAL_CAPACITY

    scope = "one process, one owner per physical device of that process"
    name = "single-process-physical-devices"
    if jax.process_count() != 1:
        return _gate(
            name, "NOT_OBSERVED", "this invocation is not one process", scope=scope
        )
    runtime = ExecutionRuntime.current()
    available = runtime.root_group.spec.device_count
    if available < 2:
        return _gate(
            name,
            "NOT_OBSERVED",
            f"one process observes {available} device; two are required",
            scope=scope,
        )
    owners = min(available, 4)
    group = _device_subset(runtime, owners)
    ownership = DistributedOwnershipPlan(owner_address(), group, LOCAL_CAPACITY)
    functional = _ownership_parities(ownership)
    errors = [
        f"{strategy}: {error}"
        for strategy, result in functional.items()
        for error in result["errors"]
    ]
    platforms = sorted({device.platform for device in group.devices})
    evidence: dict[str, Any] = {
        "owners": owners,
        "platforms": platforms,
        "device_kinds": sorted({device.device_kind for device in group.devices}),
        "forced_cpu_device_count": pxconfig.snapshot().cpu_device_count,
        "functional_parity": functional,
    }
    if errors:
        return _gate(name, "FAIL", "; ".join(errors), scope=scope, evidence=evidence)
    if pxconfig.snapshot().cpu_device_count is not None or platforms == ["cpu"]:
        return _gate(
            name,
            "NOT_OBSERVED",
            "forced or host-CPU devices passed the functional device-collective "
            "program; physical accelerators were not observed",
            scope=scope,
            evidence=evidence,
        )
    try:
        evidence["admission"] = _admission(
            runtime,
            group.spec,
            ResourceRequest(
                cpu_cores=1,
                memory_bytes=1 << 30,
                accelerator_count=owners,
                host_count=1,
                process_count=1,
            ),
            "atomistic-domain-single-process",
        )
    except ExecutionAdmissionError as error:
        return _gate(
            name,
            "NOT_OBSERVED",
            f"resource admission refused the device group: {error}",
            scope=scope,
            evidence=evidence,
        )
    return _gate(
        name,
        "PASS",
        f"{owners} admitted physical {platforms} devices match the oracle",
        scope=scope,
        evidence=evidence,
    )


# ------------------------------------------------------------ coupled launches


def _multi_process_one_host(
    directory: Path, platform: Platform, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    name = "multi-process-one-host"
    scope = "two coupled OS processes on one observed host"
    records = {
        strategy: run_scenario(
            "parity",
            directory / f"parity-{strategy}",
            devices_per_process=devices_per_process,
            strategy=strategy,
            platform=platform,
            timeout_seconds=timeout,
        )
        for strategy in STRATEGIES
    }
    errors = [
        f"{strategy}: {error}"
        for strategy, record in records.items()
        for error in record["errors"]
    ]
    if errors:
        return _gate(name, "FAIL", "; ".join(errors), scope=scope, evidence=records)
    hosts = {
        rank["runtime"]["observed_host_count"]
        for record in records.values()
        for rank in record["ranks"]
    }
    if hosts != {1}:
        return _gate(
            name,
            "NOT_OBSERVED",
            f"observed host counts {sorted(hosts, key=str)}; one Linux boot "
            "identity per process is required",
            scope=scope,
            evidence=records,
        )
    if platform == "cpu":
        scope += "; forced host-CPU devices, no accelerator claim"
    return _gate(
        name,
        "PASS",
        f"two processes on one host match the oracle ({platform} devices)",
        scope=scope,
        evidence=records,
    )


def _killed_rank(directory: Path, timeout: float) -> dict[str, Any]:
    records = {
        scenario: run_scenario(
            scenario,
            directory / scenario,
            devices_per_process=1,
            timeout_seconds=timeout,
        )
        for scenario in ("kill", "transport-kill")
    }
    errors = [
        f"{scenario}: {error}"
        for scenario, record in records.items()
        for error in record["errors"]
    ]
    return _gate(
        "killed-rank-fail-coupled",
        "FAIL" if errors else "PASS",
        "; ".join(errors)
        or "the coupled job failed before its deadline without a global commit",
        scope="launcher fail-coupling on one host; no collective rollback claim",
        evidence=records,
    )


def _same_topology(
    directory: Path, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    root = directory / "same-topology"
    published = run_scenario(
        "publish", root, devices_per_process=devices_per_process, timeout_seconds=timeout
    )
    restored = run_scenario(
        "restore", root, devices_per_process=devices_per_process, timeout_seconds=timeout
    )
    errors = published["errors"] + restored["errors"]
    if published["passed"] and restored["passed"]:
        errors.extend(compare_digests(published, restored))
    return _gate(
        "same-topology-restart",
        "FAIL" if errors else "PASS",
        "; ".join(errors) or "every continuation leaf restored bitwise",
        scope="same plan, owners and processes; bitwise leaves",
        evidence={"publish": published, "restore": restored},
    )


def _changed_topology(
    directory: Path, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    root = directory / "changed-topology"
    target = 1 if devices_per_process > 1 else 2
    published = run_scenario(
        "publish", root, devices_per_process=devices_per_process, timeout_seconds=timeout
    )
    migrated = run_scenario(
        "migrate", root, devices_per_process=target, timeout_seconds=timeout
    )
    errors = published["errors"] + migrated["errors"]
    if published["passed"] and migrated["passed"]:
        restored_rows = sum(
            rank["migration"]["restored_rows"] for rank in migrated["ranks"]
        )
        atoms = sum(published["ranks"][0]["parity"]["owned_rows_by_process"])
        if restored_rows != atoms:
            errors.append(f"restored {restored_rows} of {atoms} source rows")
        if any(
            before["digest"]["rng_key"] != after["digest"]["rng_key"]
            for before, after in zip(published["ranks"], migrated["ranks"], strict=True)
        ):
            errors.append("the RNG lineage changed across the topology change")
    return _gate(
        "changed-topology-restart",
        "FAIL" if errors else "PASS",
        "; ".join(errors)
        or f"{2 * devices_per_process} owners continued on {2 * target} owners",
        scope="changed owner topology; tolerance-class, not bitwise trajectories",
        evidence={"publish": published, "migrate": migrated},
    )


def _stage_transport(directory: Path, timeout: float) -> dict[str, Any]:
    record = run_scenario(
        "transport",
        directory / "transport",
        devices_per_process=1,
        timeout_seconds=timeout,
    )
    return _gate(
        "stage-transport-subgroup",
        "PASS" if record["passed"] else "FAIL",
        "; ".join(record["errors"]) or "subgroup transport committed and restored",
        scope="two-process child stage groups on one host",
        evidence=record,
    )


def _stage_domain_campaign(
    directory: Path, platform: Platform, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    """Backpressured two-stage domain campaign: reference, kill, restart, migrate."""
    root = directory / "stage-domain-campaign"
    target = 1 if devices_per_process > 1 else 2
    records: dict[str, Any] = {}
    for scenario, devices in (
        ("domain-campaign", devices_per_process),
        ("domain-campaign-kill", devices_per_process),
        ("domain-campaign-restart", devices_per_process),
        ("domain-campaign-publish", devices_per_process),
        ("domain-campaign-migrate", target),
    ):
        records[scenario] = run_scenario(
            scenario,
            root,
            devices_per_process=devices,
            platform=platform,
            timeout_seconds=timeout,
        )
    errors = [
        f"{scenario}: {error}"
        for scenario, record in records.items()
        for error in record["errors"]
    ]
    return _gate(
        "stage-domain-campaign",
        "FAIL" if errors else "PASS",
        "; ".join(errors)
        or "credit/sink backpressure parked finished tasks without physics; the "
        "killed rank left no commit; restart replayed bitwise; "
        f"{2 * devices_per_process}-device campaign migrated to "
        f"{2 * target} devices within tolerance",
        scope="two process-subset stage groups on one host; native MACE domain "
        "dynamics; functional collective program, not accelerator evidence",
        evidence=records,
    )


def _data_domain_training(
    directory: Path, platform: Platform, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    """Device data × domain training with both axes > 1 and a durable restart."""
    root = directory / "data-domain-training"
    records = {
        scenario: run_scenario(
            scenario,
            root,
            devices_per_process=devices_per_process,
            platform=platform,
            timeout_seconds=timeout,
        )
        for scenario in ("dp-domain", "dp-domain-restart")
    }
    errors = [
        f"{scenario}: {error}"
        for scenario, record in records.items()
        for error in record["errors"]
    ]
    return _gate(
        "data-domain-training",
        "FAIL" if errors else "PASS",
        "; ".join(errors)
        or "data and owner axes > 1 matched the independent single-device "
        "objective/gradient reference and restarted bitwise",
        scope="device mesh (data, owner) over coupled local processes; "
        "roundoff-bounded, not partition-bitwise arithmetic",
        evidence=records,
    )


def _model_family_domains(
    directory: Path, platform: Platform, devices_per_process: int, timeout: float
) -> dict[str, Any]:
    """AIMNet2/UMA/MACE-LES/LJ-PME routes and dynamic-cell trajectories."""
    from tools._atomistic_smoke_families import SCENARIO_FAMILIES

    root = directory / "model-family-domains"
    records: dict[str, Any] = {}
    for scenario, families in SCENARIO_FAMILIES.items():
        for family in families:
            for strategy in STRATEGIES:
                records[f"{scenario}:{family}:{strategy}"] = run_scenario(
                    scenario,
                    root / f"{scenario}-{family}-{strategy}",
                    devices_per_process=devices_per_process,
                    strategy=strategy,
                    platform=platform,
                    timeout_seconds=timeout,
                    family=family,
                )
    errors = [
        f"{case}: {error}"
        for case, record in records.items()
        for error in record["errors"]
    ]
    return _gate(
        "model-family-domains",
        "FAIL" if errors else "PASS",
        "; ".join(errors)
        or "every declared route matched its independent single-device program "
        "and every undeclared route was natively refused",
        scope="randomly initialized native models; no pretrained-weight, upstream "
        "or ensemble claim",
        evidence=records,
    )


# ----------------------------------------------------------- observed job


def _observed_multi_host(directory: Path) -> dict[str, Any]:
    """Every rank of an externally launched job runs this collectively."""
    name = "multi-host"
    scope = "two or more observed physical hosts in the live runtime"
    runtime = ExecutionRuntime.current()
    topology = runtime_record(runtime)
    hosts = topology["observed_host_count"]
    evidence: dict[str, Any] = {"runtime": topology}
    if jax.process_count() < 2 or hosts is None or hosts < 2:
        return _gate(
            name,
            "NOT_OBSERVED",
            f"observed {jax.process_count()} process(es) on "
            f"{'unknown' if hosts is None else hosts} host(s)",
            scope=scope,
            evidence=evidence,
        )
    try:
        evidence["admission"] = _admission(
            runtime,
            runtime.root_group.spec,
            ResourceRequest(
                cpu_cores=1,
                memory_bytes=1 << 30,
                host_count=2,
                process_count=jax.process_count(),
            ),
            "atomistic-domain-multi-host",
        )
    except ExecutionAdmissionError as error:
        return _gate(
            name,
            "NOT_OBSERVED",
            f"host/process admission refused: {error}",
            scope=scope,
            evidence=evidence,
        )
    ranks = {strategy: run_rank("parity", directory, strategy) for strategy in STRATEGIES}
    evidence["parity"] = ranks
    local = all(record["passed"] for record in ranks.values())
    everywhere = np.asarray(
        multihost_utils.process_allgather(np.asarray(local, dtype=np.bool_), tiled=False)
    )
    if not bool(np.all(everywhere)):
        return _gate(
            name,
            "FAIL",
            "at least one rank failed oracle parity",
            scope=scope,
            evidence=evidence,
        )
    return _gate(
        name,
        "PASS",
        f"{jax.process_count()} processes on {hosts} hosts match the oracle",
        scope=scope
        + f"; device platforms {topology['platforms']}, no accelerator claim implied",
        evidence=evidence,
    )


def qualify(
    directory: Path,
    /,
    *,
    gates: Sequence[str] = GATES,
    platform: Platform = "cpu",
    devices_per_process: int = 2,
    timeout_seconds: float = 900.0,
    observe_current_job: bool = False,
) -> dict[str, Any]:
    """Evaluate the requested gates; unrequested gates are NOT_OBSERVED."""
    unknown = sorted(set(gates) - set(GATES))
    if unknown:
        raise ValueError(f"Unknown gates {unknown}.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    if observe_current_job:
        if "multi-host" in gates:
            results["multi-host"] = _guarded(
                "multi-host",
                "observed multi-host job",
                lambda: _observed_multi_host(directory / "observed"),
            )
    else:
        local: dict[str, tuple[str, Callable[[], dict[str, Any]]]] = {
            "lane-reference": ("lane reference", _lane_reference),
            "single-process-physical-devices": (
                "one process",
                _single_process_physical_devices,
            ),
            "multi-process-one-host": (
                "local launch",
                lambda: _multi_process_one_host(
                    directory, platform, devices_per_process, timeout_seconds
                ),
            ),
            "killed-rank-fail-coupled": (
                "local launch",
                lambda: _killed_rank(directory, timeout_seconds),
            ),
            "same-topology-restart": (
                "local launch",
                lambda: _same_topology(directory, devices_per_process, timeout_seconds),
            ),
            "changed-topology-restart": (
                "local launch",
                lambda: _changed_topology(
                    directory, devices_per_process, timeout_seconds
                ),
            ),
            "stage-transport-subgroup": (
                "local launch",
                lambda: _stage_transport(directory, timeout_seconds),
            ),
            "stage-domain-campaign": (
                "local launch",
                lambda: _stage_domain_campaign(
                    directory, platform, devices_per_process, timeout_seconds
                ),
            ),
            "data-domain-training": (
                "local launch",
                lambda: _data_domain_training(
                    directory, platform, devices_per_process, timeout_seconds
                ),
            ),
            "model-family-domains": (
                "local launch",
                lambda: _model_family_domains(
                    directory, platform, devices_per_process, timeout_seconds
                ),
            ),
        }
        for name, (scope, check) in local.items():
            if name in gates:
                results[name] = _guarded(name, scope, check)
    ordered = []
    for name in GATES:
        if name in results:
            ordered.append(results[name])
        elif name == "multi-host" and not observe_current_job:
            ordered.append(
                _gate(
                    name,
                    "NOT_OBSERVED",
                    "launch_local_processes couples processes on one host; two or "
                    "more hosts are observed only with --observe-current-job on "
                    "every rank of an externally launched multi-host job",
                    scope="two or more observed physical hosts",
                )
            )
        else:
            ordered.append(
                _gate(
                    name,
                    "NOT_OBSERVED",
                    "not evaluated in this invocation",
                    scope="unevaluated",
                )
            )
    runtime = ExecutionRuntime.current()
    record: dict[str, Any] = {
        "capability": CAPABILITY,
        "mode": "observe-current-job" if observe_current_job else "local",
        "configuration": {
            "gates": list(gates),
            "platform": platform,
            "devices_per_process": devices_per_process,
            "timeout_seconds": timeout_seconds,
        },
        "gates": ordered,
        "observed": [gate["name"] for gate in ordered if gate["status"] == "PASS"],
        "failed": [gate["name"] for gate in ordered if gate["status"] == "FAIL"],
        "passed": all(gate["status"] != "FAIL" for gate in ordered),
        "nonclaims": list(NONCLAIMS),
        "runtime": runtime_record(runtime),
        "environment": capture_environment().to_dict(),
    }
    identity = capture_benchmark_identity(PROJECT_ROOT, Path(__file__), tuple(record))
    record["identity"] = identity.to_dict()
    return json_safe(record)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gates", nargs="+", choices=GATES, default=list(GATES))
    parser.add_argument("--platform", choices=get_args(Platform), default="cpu")
    parser.add_argument("--devices-per-process", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--observe-current-job",
        action="store_true",
        help="Run on every rank of an externally launched multi-process job.",
    )
    arguments = parser.parse_args(argv)
    if arguments.devices_per_process < 1 or arguments.timeout_seconds <= 0:
        parser.error("devices-per-process and timeout-seconds must be positive")
    # In-process domain ownership requires 64-bit arrays.
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    with tempfile.TemporaryDirectory(prefix="atomistic-qualification-") as scratch:
        directory = arguments.directory or Path(scratch)
        record = qualify(
            directory,
            gates=arguments.gates,
            platform=arguments.platform,
            devices_per_process=arguments.devices_per_process,
            timeout_seconds=arguments.timeout_seconds,
            observe_current_job=arguments.observe_current_job,
        )
    if arguments.output is not None and jax.process_index() == 0:
        write_json_evidence(arguments.output, record)
    summary = {gate["name"]: gate["status"] for gate in record["gates"]}
    print(json.dumps({"passed": record["passed"], "gates": summary}, sort_keys=True))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
