# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native molecular-crystal packing: preparation, compile, first/warm relief and yield.

Every case packs the same caller-declared molecular input under the same
declared sampling distribution (setting, ``Z'``, cell and relief policies) and
attempt budget, so rows differ only in the measured capacity axis (lanes,
setting/operation count, contact storage).

Phases per case: packer preparation (synchronized, with its traced host
allocation peak; tracing overhead included), stream preparation, the probe
admission of every lane through the sampler's own ``admit`` with the
campaign's ordinal-addressed proposal keys (ordinals ``0..lanes-1``, every lane
admitted even when the attempt budget is smaller), lowering and compilation
of the campaign's own lane-advance executable (``_advance_jit`` with the
relief check interval), and first and warm synchronized calls of that
compiled executable from the same admitted lanes.  Compiler argument, output,
temporary and generated-code bytes come from the retained executable; logical
bytes of the prepared packer, the compiled member and the lane states are
recorded beside the process peak RSS.

Relief activity is derived from the lane step counters and terminal status
bits of the measured call, so a call whose lanes are already terminal is
reported as no-op iterations rather than as relief physics.  The campaign
records attempts, accepted proposals, shortfall, per-status attempt counts
and the relief steps of every published attempt.  Attempts per second alone
is not a quality measure: accepted proposals still need physical relaxation
and ranking.  ``scaling`` fits log-log byte exponents against lanes per
setting.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import (
    fitted_scaling_exponent,
    host_allocation_peak,
    process_peak_rss_bytes,
)
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax.atomistic._generation import (
    _advance_jit,
    AtomisticMemoryGenerationSink,
    AtomisticProposalStatus,
    AtomisticSamplerLaneState,
    proposal_address,
    proposal_keys,
)
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticDenseImageSearch,
    AtomisticRoleCapacity,
)
from phydrax.atomistic.crystal import (
    ConformerPool,
    CrystalReliefMethod,
    MolecularComponent,
    MolecularConformer,
    MolecularCrystalInput,
    standard_space_group_setting,
    VolumePolicy,
)
from phydrax.atomistic.crystal._packing import (
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
)
from phydrax.discretization import ParticleImageCapacity


_SCALED_SERIES = (
    ("compiler_argument_bytes", ("compiler", "argument_bytes")),
    ("compiler_output_bytes", ("compiler", "output_bytes")),
    ("compiler_temporary_bytes", ("compiler", "temporary_bytes")),
    ("prepared_packer_logical_bytes", ("memory", "prepared_packer_logical_bytes")),
    ("lane_state_logical_bytes", ("memory", "lane_state_logical_bytes")),
)


def _dichloroethene() -> ConformerPool:
    component = MolecularComponent(
        (6, 6, 17, 17, 1, 1),
        ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5)),
        bond_orders=(2, 1, 1, 1, 1),
        masses=(12.011, 12.011, 35.45, 35.45, 1.008, 1.008),
        spin_multiplicity=1,
        double_bond_stereo=((2, 0, 1, 3, -1),),
        name="trans-1,2-dichloroethene",
    )
    carbon = np.asarray((0.665, 0.0, 0.0))
    chlorine = carbon + 1.73 * np.asarray((0.5, np.sqrt(0.75), 0.0))
    hydrogen = carbon + 1.08 * np.asarray((0.5, -np.sqrt(0.75), 0.0))
    positions = np.stack((carbon, -carbon, chlorine, -chlorine, hydrogen, -hydrogen))
    conformer = MolecularConformer(
        component, positions, provenance="idealized benchmark geometry"
    )
    return ConformerPool(component, (conformer,))


def _plan(number: int, lanes: int, contact: float, steps: int) -> CrystalPackingPlan:
    pool = _dichloroethene()
    atoms = pool.component.atom_count
    width = 16384
    resources = CrystalContactResources(
        AtomisticDenseImageSearch(maximum_dense_routes=lanes * 2_000_000),
        search_capacities=(
            ParticleImageCapacity(
                maximum_particles_per_cell=128,
                maximum_edges=lanes * width,
                maximum_degree=width,
                maximum_images=4096,
                maximum_case_edges=width,
            ),
        ),
        role_capacities=(
            AtomisticRoleCapacity(maximum_routes=lanes * width, maximum_degree=width),
        ),
        skin=0.5,
    )
    return CrystalPackingPlan(
        MolecularCrystalInput(
            (pool,),
            contact=np.full((atoms, atoms), contact),
            volume=VolumePolicy(formula_unit_volume=95.0),
        ),
        CrystalSpaceGroupPolicy.fixed(standard_space_group_setting(number)),
        z_prime=1,
        chirality="enantiopure",
        resources=resources,
        cell=CrystalCellPolicy(minimum_height=3.0, minimum_height_floor=2.5),
        relief=CrystalReliefPolicy(maximum_steps=steps, check_interval=10),
    )


def _relief_method(state: AtomisticSamplerLaneState, /) -> CrystalReliefMethod:
    method = state.method
    if not isinstance(method, CrystalReliefMethod):
        raise TypeError("Packing lanes must carry a CrystalReliefMethod continuation.")
    return method


def _relief_activity(
    before: AtomisticSamplerLaneState,
    after: AtomisticSamplerLaneState,
    iterations: int,
    /,
) -> dict[str, int]:
    """Relief work of one lane advance from step counters and terminal bits.

    A lane is live while running, unaccepted and unfailed.  Each live
    iteration evaluates its overlap energy and either moves (step counter + 1,
    step-budget exhaustion included) or holds terminally with an acceptance or
    a failure.  The fused loop runs a relief step while any lane is live, so
    executed iterations are the largest per-lane live count; the remaining
    loop iterations are no-ops that evaluate nothing.
    """
    initial = _relief_method(before)
    final = _relief_method(after)
    running = np.asarray(before.active) & ~np.asarray(before.terminal)
    live = running & (np.asarray(initial.status) == 0) & ~np.asarray(initial.accepted)
    moves = np.asarray(final.steps, dtype=np.int64) - np.asarray(
        initial.steps, dtype=np.int64
    )
    status = np.asarray(final.status)
    accepted = np.asarray(final.accepted)
    exhausted_bit = int(AtomisticProposalStatus.STEP_BUDGET_EXHAUSTED)
    exhausted = live & (status == exhausted_bit)
    held = live & (accepted | ((status != 0) & (status != exhausted_bit)))
    evaluations = np.where(live, moves + held.astype(np.int64), 0)
    executed = int(evaluations.max(initial=0))
    lanes = int(running.shape[0])
    return {
        "lanes": lanes,
        "running_lanes": int(np.count_nonzero(running)),
        "live_lanes": int(np.count_nonzero(live)),
        "live_lane_evaluations": int(evaluations.sum()),
        "relief_moves": int(np.where(live, moves, 0).sum()),
        "executed_iterations": executed,
        "noop_iterations": iterations - executed,
        "evaluated_lane_slots": executed * lanes,
        "accepted_lanes": int(np.count_nonzero(live & accepted)),
        "failed_lanes": int(np.count_nonzero(held & ~accepted)),
        "exhausted_lanes": int(np.count_nonzero(exhausted)),
        "terminal_lanes_after": int(np.count_nonzero(np.asarray(after.terminal))),
    }


def _case(
    number: int, lanes: int, budget: int, repeats: int, contact: float, steps: int
) -> dict[str, Any]:
    plan = _plan(number, lanes, contact, steps)
    interval = plan.relief.check_interval
    (packer, prepare_peak), prepare_seconds = measure_host(
        lambda: host_allocation_peak(lambda: synchronize(plan.prepare(lanes=lanes)))
    )
    campaign = f"benchmark-{number}-{lanes}"
    runtime, stream_seconds = measure_synchronized(
        lambda: packer.stream(campaign).prepare()
    )
    member = packer.members[0]
    ordinals = jnp.arange(lanes, dtype=jnp.int64)
    keys = proposal_keys(
        jr.key(0),
        proposal_address(
            campaign, member.support.support_id, member.sampler_id, role="proposal"
        ),
        ordinals,
    )
    admitted, admission_seconds = measure_synchronized(
        lambda: member.admit(
            member.vacant(lanes),
            jnp.ones((lanes,), dtype=jnp.bool_),
            keys,
            ordinals,
            None,
        )
    )
    compiled, compile_timing = measure_lower_and_compile(
        lambda: _advance_jit.lower(  # ty: ignore[unresolved-attribute]
            member, admitted, interval
        ),
        lambda lowered: lowered.compile(),
    )
    cost = compiled.compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    compiler = compiler_evidence(cost, compiled.compiled.memory_analysis(), source="xla")
    advanced, first = measure_synchronized(lambda: compiled(member, admitted, interval))
    repeated, warm = measure_repeated(
        lambda: compiled(member, admitted, interval), warmup=1, repeats=repeats
    )
    first_activity = _relief_activity(admitted, advanced, interval)
    warm_activity = _relief_activity(admitted, repeated, interval)
    sink = AtomisticMemoryGenerationSink("benchmark")
    start = runtime.start(jr.key(1), target=budget, attempt_budget=budget)
    transitions = (budget // lanes + 2) * (steps // interval + 2)
    final, campaign_seconds = measure_synchronized(
        lambda: runtime.run(start, sink, maximum_transitions=transitions)
    )
    statuses = Counter(
        "accepted"
        if record.status == 0
        else "|".join(
            flag.name for flag in AtomisticProposalStatus if flag and record.status & flag
        )
        for record in sink.records
    )
    return {
        "international_number": number,
        "operations": member.layout.operation_count,
        "atoms": member.layout.atom_count,
        "lanes": lanes,
        "attempt_budget": budget,
        "relief_check_interval": interval,
        "relief_maximum_steps": steps,
        "phases": {
            "prepare": {
                "seconds": prepare_seconds,
                "host_allocation_peak_bytes": prepare_peak,
                "tracing_included": True,
            },
            "stream_prepare_seconds": stream_seconds,
            "probe_admission_seconds": admission_seconds,
            "compile": asdict(compile_timing),
            "first_transition_seconds": first,
            "warm_transition": warm.to_dict(unit="seconds"),
        },
        "compiler": asdict(compiler),
        "relief_activity": {
            "first": first_activity,
            "warm": warm_activity,
            "warm_matches_first": warm_activity == first_activity,
        },
        "memory": {
            "prepared_packer_logical_bytes": logical_array_bytes(packer),
            "member_logical_bytes": logical_array_bytes(member),
            "lane_state_logical_bytes": logical_array_bytes(admitted),
            "advanced_lane_state_logical_bytes": logical_array_bytes(advanced),
            "campaign_final_state_logical_bytes": logical_array_bytes(final),
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        },
        "campaign": {
            "seconds": campaign_seconds,
            "transitions": int(final.cursor.transitions),
            "attempts": len(sink.records),
            "accepted": statuses.get("accepted", 0),
            "shortfall": runtime.shortfall(final),
            "stop_reason": final.stop_reason.name,
            "statuses": dict(statuses),
            "relief_steps": sum(int(record.value("steps")) for record in sink.records),
        },
    }


def _series_value(case: dict[str, Any], path: tuple[str, str], /) -> int | None:
    section, name = path
    value = case[section][name]
    return None if value is None else int(value)


def _scaling(cases: list[dict[str, Any]], /) -> dict[str, Any]:
    """Log-log byte exponents against lanes for every setting with two capacities."""
    result: dict[str, Any] = {}
    for number in sorted({case["international_number"] for case in cases}):
        rows = sorted(
            (case for case in cases if case["international_number"] == number),
            key=lambda case: case["lanes"],
        )
        capacities = [row["lanes"] for row in rows]
        series: dict[str, Any] = {}
        for name, path in _SCALED_SERIES:
            values = [_series_value(row, path) for row in rows]
            known = [value for value in values if value is not None]
            reason = None
            if len(capacities) < 2:
                reason = "fewer than two lane capacities"
            elif len(known) != len(values):
                reason = "compiler memory analysis unavailable"
            elif 0 in known:
                reason = "a zero byte count has no log-log exponent"
            series[name] = {
                "bytes": values,
                "exponent": (
                    None
                    if reason is not None
                    else fitted_scaling_exponent(capacities, known)
                ),
                "unavailable_reason": reason,
            }
        result[str(number)] = {"lanes": capacities, "series": series}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 2, 14])
    parser.add_argument("--lanes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--contact", type=float, default=3.2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.budget < 1
        or args.repeats < 1
        or args.steps < 10
        or any(lane < 1 for lane in args.lanes)
    ):
        raise ValueError(
            "Budgets, repeats and lanes must be positive; steps at least 10."
        )
    if list(args.lanes) != sorted(set(args.lanes)):
        raise ValueError("Lanes must be strictly increasing capacities.")
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/atomistic/_generation.py",
        "phydrax/atomistic/crystal/_packing.py",
        "phydrax/atomistic/crystal/_asu.py",
        "phydrax/atomistic/_neighborhood_bundle.py",
    )
    cases = [
        _case(number, lanes, args.budget, args.repeats, args.contact, args.steps)
        for number in args.groups
        for lanes in args.lanes
    ]
    record = {
        "benchmark": "native-crystal-packing",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root, Path(__file__), ("cases", "environment", "scaling", "source_sha256")
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "cases": cases,
        "scaling": _scaling(cases),
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
