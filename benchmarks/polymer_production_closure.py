from __future__ import annotations

import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment, logical_array_bytes, measure_repeated
from phydrax.applications.polymer_liquids import entanglement as ent, reptation as rep


def main() -> int:
    particle_count = 128
    system = phx.atomistic.AtomisticSystemPlan(
        np.arange(particle_count),
        np.zeros((particle_count,), dtype=np.int32),
        np.ones((particle_count,)),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros((particle_count,), dtype=np.int32),
        element_mask=np.zeros((particle_count,), dtype=bool),
    ).prepare()
    mobility = phx.atomistic.FreeSpaceRPYMobilityPlan(
        0.5, 1.0, maximum_particles=particle_count
    ).prepare(system, np.arange(particle_count))
    coordinate = jnp.stack(
        (
            jnp.arange(particle_count, dtype=float) * 1.25,
            jnp.zeros((particle_count,)),
            jnp.zeros((particle_count,)),
        ),
        axis=-1,
    )
    force = jnp.ones_like(coordinate)
    rpy_operation = eqx.filter_jit(
        lambda positions, values: mobility.operator(positions).mv(values)
    )
    rpy_result, rpy_timing = measure_repeated(
        lambda: rpy_operation(coordinate, force), warmup=1, repeats=3
    )

    glamm = rep.GLAMMPlan(
        128,
        1.0e-5,
        1.0,
        100.0,
        10.0,
        contour_diffusivity=0.01,
    )
    glamm_state = glamm.initialize()
    gradient = jnp.asarray([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    glamm_operation = eqx.filter_jit(rep.glamm_step)
    glamm_result, glamm_timing = measure_repeated(
        lambda: glamm_operation(glamm, glamm_state, gradient), warmup=1, repeats=3
    )

    estimator = ent.EntanglementEstimatorPlan(
        0.85,
        1.0,
        block_count=8,
        minimum_frames=64,
        maximum_relative_standard_error=1.0,
    )
    end_to_end = jnp.full((256, 64), 100.0)
    contour = jnp.full((256, 64), 40.0)
    estimator_operation = eqx.filter_jit(ent.estimate_entanglement)
    estimator_result, estimator_timing = measure_repeated(
        lambda: estimator_operation(estimator, jnp.full((64,), 100), end_to_end, contour),
        warmup=1,
        repeats=3,
    )

    successful = bool(
        jnp.all(jnp.isfinite(rpy_result))
        & glamm_result.successful
        & estimator_result.successful
    )
    payload = {
        "benchmark": "polymer-production-closure",
        "free_space_rpy": {
            "particle_count": particle_count,
            "timing": rpy_timing.to_milliseconds_dict(),
            "logical_bytes": logical_array_bytes(rpy_result),
        },
        "glamm": {
            "contour_nodes": glamm.contour_nodes,
            "timing": glamm_timing.to_milliseconds_dict(),
            "logical_bytes": logical_array_bytes(glamm_result),
        },
        "entanglement_estimator": {
            "frames": 256,
            "chains": 64,
            "timing": estimator_timing.to_milliseconds_dict(),
            "logical_bytes": logical_array_bytes(estimator_result),
        },
        "successful": successful,
        "environment": capture_environment().to_dict(),
        "release_claim": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
