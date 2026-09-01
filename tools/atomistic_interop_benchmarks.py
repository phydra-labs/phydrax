import json
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


with tempfile.TemporaryDirectory(prefix="phydrax-interop-") as directory:
    path = Path(directory) / "trajectory.h5"
    plan = phx.atomistic.interchange.H5MDTrajectoryPlan(path)
    frame = phx.atomistic.AtomisticFrame(
        0.0,
        0,
        jnp.zeros((128, 3)),
        jnp.arange(128),
        system_id="benchmark-system",
        topology_id="benchmark-topology",
        unit_system_id="benchmark-units",
        source_id="benchmark-frame",
    )
    started = time.perf_counter()
    with plan.open(append=False) as writer:
        for index in range(100):
            writer.write(
                phx.atomistic.AtomisticFrame(
                    index * 0.001,
                    index,
                    frame.positions,
                    frame.stable_ids,
                    system_id=frame.system_id,
                    topology_id=frame.topology_id,
                    unit_system_id=frame.unit_system_id,
                    source_id=f"benchmark-frame-{index}",
                )
            )
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with plan.open() as reader:
        frames = tuple(reader)
    read_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "frames": len(frames),
                "bytes": path.stat().st_size,
                "write_frames_per_second": len(frames) / write_seconds,
                "read_frames_per_second": len(frames) / read_seconds,
                "roundtrip": bool(jnp.array_equal(frames[-1].positions, frame.positions)),
            },
            indent=2,
            sort_keys=True,
        )
    )
