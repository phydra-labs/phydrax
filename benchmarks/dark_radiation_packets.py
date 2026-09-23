#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import ADMGridGeometry, RelativityConvention
from phydrax.solver._dark_radiation_packets import DarkRadiationPacketPlan
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan


def _frame(units, *, time_value, scale_factor, token):
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(token),
        chart_id="packet-benchmark",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="packet-benchmark-domain",
        geometry_lineage_id="packet-benchmark-flrw",
    )
    return LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        jnp.zeros((4,)),
        jnp.asarray(time_value),
        jnp.asarray(scale_factor),
        observer_id="packet-benchmark-observer",
        orientation_id="right-handed-future",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=int, default=65536)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.packets <= 0
        or arguments.groups <= 0
        or arguments.repetitions <= 0
        or arguments.shards <= 0
    ):
        raise ValueError("Packet, group, repetition, and shard counts must be positive.")

    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    frame0 = _frame(units, time_value=0.0, scale_factor=1.0, token=1)
    frame1 = _frame(units, time_value=0.01, scale_factor=1.001, token=2)
    epoch = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=arguments.packets,
        product_capacity=1,
        radiation_capacity=arguments.packets,
        work_capacity=1,
        frontier_capacity=1,
        packet_width=1,
        event_width=1,
        product_width=1,
        radiation_width=20,
        work_width=1,
        frontier_width=1,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
        shard_count=arguments.shards,
    )
    edges = jnp.geomspace(0.5, 2.0, arguments.groups + 1)
    plan = DarkRadiationPacketPlan(
        arguments.packets,
        arguments.packets,
        units,
        edges,
        jnp.asarray((-100.0, -100.0, -100.0)),
        jnp.asarray((100.0, 100.0, 100.0)),
        epoch_plan=epoch,
    )
    count = arguments.packets
    identifiers = jnp.arange(count, dtype=jnp.int64) + 1
    directions = jr.normal(jr.key(3), (count, 3))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    energy = jnp.ones((count, 1))
    four = jnp.concatenate((energy, energy * directions), axis=-1)
    admitted = plan.admit(
        plan.empty(frame0, epoch_manifest_id="a" * 64),
        identifiers,
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.stack((2 * identifiers, 2 * identifiers + 1), axis=-1),
        identifiers + 10_000_000,
        jnp.zeros((count,), dtype=jnp.int32),
        jnp.zeros((count, 3)),
        four,
        jnp.full((count,), arguments.groups // 2, dtype=jnp.int32),
        jnp.broadcast_to(jnp.asarray((1.0, 0.1, 0.0, 0.0)), (count, 4)),
        jnp.full((count,), 0.5),
        jr.split(jr.key(5), count),
        jnp.arange(count, dtype=jnp.int32) % arguments.shards,
        jnp.ones((count,)),
        jnp.ones((count,), dtype="bool"),
    )
    if not bool(admitted.successful):
        raise RuntimeError("Packet benchmark admission failed.")
    advance = eqx.filter_jit(plan.advance)
    opacity = jnp.full((count,), 0.1)
    mueller = jnp.diag(jnp.asarray((1.0, 0.8, 0.8, 0.8)))
    keyword = {
        "next_epoch_manifest_id": "b" * 64,
        "end_frame_realization_id": frame1.realization_id(),
    }
    start = time.perf_counter()
    first = advance(
        admitted.accepted_state,
        frame0,
        frame1,
        jnp.zeros_like(opacity),
        opacity,
        mueller,
        **keyword,
    )
    jax.block_until_ready(first)
    compile_and_first = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(arguments.repetitions):
        result = advance(
            admitted.accepted_state,
            frame0,
            frame1,
            jnp.zeros_like(opacity),
            opacity,
            mueller,
            **keyword,
        )
    jax.block_until_ready(result)
    execution = (time.perf_counter() - start) / arguments.repetitions
    if not bool(result.successful):
        raise RuntimeError("Packet benchmark transport failed.")
    epoch_state = plan.to_dark_sector_epoch_state(
        result.accepted_state, parent_epoch_manifest_id="a" * 64
    )
    payload = {
        "packets": count,
        "groups": arguments.groups,
        "collectives": arguments.shards,
        "events": int(result.evidence.event_count),
        "compile_and_first_ms": 1000.0 * compile_and_first,
        "execution_ms": 1000.0 * execution,
        "packets_per_second": count / execution,
        "four_force_residual": float(
            jnp.max(jnp.abs(result.evidence.four_force_residual))
        ),
        "mass_shell_residual": float(
            jnp.max(jnp.abs(result.evidence.mass_shell_residual))
        ),
        "epoch_conservation_residual": float(
            jnp.max(jnp.abs(epoch_state.conservation_residual))
        ),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
