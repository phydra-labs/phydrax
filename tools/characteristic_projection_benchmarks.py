#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import diffrax as dfx
import jax
import jax.numpy as jnp

import phydrax as phx


def _rotation_velocity(_time, points, _args):
    return jnp.stack((-points[..., 1], points[..., 0]), axis=-1)


def _exact_foot(points, duration):
    cosine = jnp.cos(duration)
    sine = jnp.sin(duration)
    x = cosine * points[..., 0] + sine * points[..., 1]
    y = -sine * points[..., 0] + cosine * points[..., 1]
    return jnp.stack((x, y), axis=-1)


def _run(solver_name: str, count: int) -> dict[str, object]:
    angles = 2.0 * jnp.pi * jnp.arange(count) / count
    points = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
    selected = dfx.Tsit5() if solver_name == "tsit5" else dfx.Dopri5()
    started = time.perf_counter()
    result = phx.solver.trace_characteristics(
        _rotation_velocity,
        points,
        0.0,
        1.0,
        solver=selected,
        rtol=1e-8,
        atol=1e-10,
    )
    jax.block_until_ready(result.foot_points)
    elapsed = time.perf_counter() - started
    error = result.foot_points - _exact_foot(points, 1.0)
    return {
        "solver": solver_name,
        "point_count": count,
        "successful": bool(result.successful),
        "max_error": float(jnp.max(jnp.abs(error))),
        "accepted_steps": int(result.solution.stats["num_accepted_steps"]),
        "rejected_steps": int(result.solution.stats["num_rejected_steps"]),
        "wall_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--point-count", type=int, default=256)
    args = parser.parse_args()
    rows = [_run(name, args.point_count) for name in ("tsit5", "dopri5")]
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
