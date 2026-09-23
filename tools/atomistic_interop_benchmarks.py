import json
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


units = phx.atomistic.AtomisticUnitSystem.reduced()

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
        units=units,
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
                    units=frame.units,
                    source_id=f"benchmark-frame-{index}",
                )
            )
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with plan.open() as reader:
        frames = tuple(reader)
    read_seconds = time.perf_counter() - started
    roundtrip = len(frames) == 100 and all(
        bool(
            jnp.asarray(observed.time) == index * 0.001
            and jnp.asarray(observed.step) == index
            and jnp.array_equal(observed.positions, frame.positions)
            and jnp.array_equal(observed.stable_ids, frame.stable_ids)
            and observed.velocities is None
            and observed.momenta is None
            and observed.forces is None
            and observed.cell_vectors is None
            and observed.image_counts is None
            and observed.energy is None
            and not observed.auxiliary
            and bool(observed.valid)
            and observed.coordinate_domain == frame.coordinate_domain
            and observed.system_id == frame.system_id
            and observed.topology_id == frame.topology_id
            and observed.units.unit_system_id == frame.units.unit_system_id
            and observed.source_id == f"benchmark-frame-{index}"
        )
        for index, observed in enumerate(frames)
    )
    print(
        json.dumps(
            {
                "frames": len(frames),
                "bytes": path.stat().st_size,
                "write_frames_per_second": len(frames) / write_seconds,
                "read_frames_per_second": len(frames) / read_seconds,
                "roundtrip": roundtrip,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not roundtrip:
        raise SystemExit(1)
