#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure source-2014 batched constitutive stress and native mixed FE residual."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.applications.skeletal_muscle.continuum import (
    HeidlaufRoehrle2014Parameters,
    HeidlaufRoehrle2014Plan,
    HeidlaufRoehrle2014StressInput,
    UniformFiberArchitecturePlan,
)
from phydrax.discretization import (
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def _time(function, value, repetitions):
    started = time.perf_counter()
    result = function(value)
    jax.block_until_ready(result)
    first = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repetitions):
        result = function(value)
    jax.block_until_ready(result)
    return result, first, (time.perf_counter() - started) / repetitions


def run(points, repetitions):
    parameters = HeidlaufRoehrle2014Parameters.published_table_2()
    material = HeidlaufRoehrle2014Plan(
        "benchmark-2014", "benchmark-prescribed-stress"
    ).prepare(
        parameters,
        UniformFiberArchitecturePlan("benchmark-2014-x").prepare(
            jnp.asarray((1.0, 0.0, 0.0))
        ),
        HeidlaufRoehrle2014StressInput(
            0.65, jnp.zeros(8, dtype=jnp.uint32), "benchmark-prescribed-stress"
        ),
    )
    stretches = jnp.linspace(0.8, 1.3, points)
    deformations = jax.vmap(
        lambda stretch: jnp.diag(jnp.asarray((stretch, stretch**-0.5, stretch**-0.5)))
    )(stretches)
    pressure = jnp.full((points,), 1000.0)
    gamma = jnp.linspace(0.0, 1.6, points)
    constitutive = eqx.filter_jit(
        lambda deformation: material.first_piola_points(deformation, pressure, gamma)
    )
    stresses, point_first, point_steady = _time(constitutive, deformations, repetitions)
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    origin = 2 * parameters.c10_pa + 4 * parameters.c01_pa
    mixed = material.prepare_qualified_mixed(
        MixedFiniteElementConstraintPlan(mesh, PressureGaugePolicy("mean-zero")),
        pressure_origin_pa=origin,
    )
    residual, fe_first, fe_steady = _time(
        eqx.filter_jit(mixed.problem.residual),
        mixed.problem.state_space.zeros(),
        repetitions,
    )
    finite = bool(jnp.all(jnp.isfinite(stresses))) and all(
        bool(jnp.all(jnp.isfinite(value))) for value in residual
    )
    return {
        "source": "doi:10.3389/fphys.2014.00498; prescribed-stress material only",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "dtype": str(stresses.dtype),
        "points": points,
        "repetitions": repetitions,
        "constitutive_first_execution_seconds": point_first,
        "constitutive_steady_seconds": point_steady,
        "material_points_per_second": points / point_steady,
        "native_fe_cell_count": 1,
        "native_fe_displacement_dofs": int(mixed.problem.state_space.zeros()[0].size),
        "native_fe_pressure_dofs": int(mixed.problem.state_space.zeros()[1].size),
        "native_fe_first_execution_seconds": fe_first,
        "native_fe_steady_seconds": fe_steady,
        "stress_checksum_pa": float(jnp.sum(stresses)),
        "finite": finite,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.points <= 0 or args.repetitions <= 0:
        parser.error("points and repetitions must be positive")
    report = run(32 if args.smoke else args.points, 2 if args.smoke else args.repetitions)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if report["finite"] else 1)


if __name__ == "__main__":
    main()
